import os
from os.path import join
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyspark.sql.functions as sf

from lightfm import LightFM
from pyspark.sql import DataFrame
from scipy.sparse import csr_matrix, hstack, diags
from sklearn.preprocessing import MinMaxScaler

from replay.preprocessing import CSRConverter
from replay.data import REC_SCHEMA
from replay.experimental.models.base_rec import HybridRecommender
from replay.utils.spark_utils import (
    check_numeric,
    save_picklable_to_parquet,
    load_pickled_from_parquet,
)
from replay.utils.session_handler import State


# pylint: disable=too-many-locals, too-many-instance-attributes
class LightFMWrap(HybridRecommender):
    """Wrapper for LightFM."""

    epochs: int = 10
    _search_space = {
        "loss": {
            "type": "categorical",
            "args": ["logistic", "bpr", "warp", "warp-kos"],
        },
        "no_components": {"type": "loguniform_int", "args": [8, 512]},
    }
    user_feat_scaler: Optional[MinMaxScaler] = None
    item_feat_scaler: Optional[MinMaxScaler] = None

    def __init__(
        self,
        no_components: int = 128,
        loss: str = "warp",
        random_state: Optional[int] = None,
    ):  # pylint: disable=too-many-arguments
        np.random.seed(42)
        self.no_components = no_components
        self.loss = loss
        self.random_state = random_state
        cpu_count = os.cpu_count()
        self.num_threads = cpu_count if cpu_count is not None else 1

    @property
    def _init_args(self):
        return {
            "no_components": self.no_components,
            "loss": self.loss,
            "random_state": self.random_state,
        }

    def _save_model(self, path: str):
        save_picklable_to_parquet(self.model, join(path, "model"))
        save_picklable_to_parquet(self.user_feat_scaler, join(path, "user_feat_scaler"))
        save_picklable_to_parquet(self.item_feat_scaler, join(path, "item_feat_scaler"))

    def _load_model(self, path: str):
        self.model = load_pickled_from_parquet(join(path, "model"))
        self.user_feat_scaler = load_pickled_from_parquet(join(path, "user_feat_scaler"))
        self.item_feat_scaler = load_pickled_from_parquet(join(path, "item_feat_scaler"))

    def _feature_table_to_csr(
        self,
        log_ids_list: DataFrame,
        feature_table: Optional[DataFrame] = None,
    ) -> Optional[csr_matrix]:
        """
        Transform features to sparse matrix
        Matrix consists of two parts:
        1) Left one is a ohe-hot encoding of user and item ids.
        Matrix size is: number of users or items * number of user or items in fit.
        Cold users and items are represented with empty strings
        2) Right one is a numerical features, passed with feature_table.
        MinMaxScaler is applied per column, and then value is divided by the row sum.

        :param feature_table: dataframe with ``user_idx`` or ``item_idx``,
            other columns are features.
        :param log_ids_list: dataframe with ``user_idx`` or ``item_idx``,
            containing unique ids from log.
        :returns: feature matrix
        """

        if feature_table is None:
            return None

        check_numeric(feature_table)
        log_ids_list = log_ids_list.distinct()
        entity = "item" if "item_idx" in feature_table.columns else "user"
        idx_col_name = f"{entity}_idx"

        # filter features by log
        feature_table = feature_table.join(
            log_ids_list, on=idx_col_name, how="inner"
        )

        fit_dim = getattr(self, f"_{entity}_dim")
        matrix_height = max(
            fit_dim,
            log_ids_list.select(sf.max(idx_col_name)).collect()[0][0] + 1,
        )
        if not feature_table.rdd.isEmpty():
            matrix_height = max(
                matrix_height,
                feature_table.select(sf.max(idx_col_name)).collect()[0][0] + 1,
            )

        features_np = (
            feature_table.select(
                idx_col_name,
                # first column contains id, next contain features
                *(
                    sorted(
                        list(
                            set(feature_table.columns).difference(
                                {idx_col_name}
                            )
                        )
                    )
                ),
            )
            .toPandas()
            .to_numpy()
        )
        entities_ids = features_np[:, 0]
        features_np = features_np[:, 1:]
        number_of_features = features_np.shape[1]

        all_ids_list = log_ids_list.toPandas().to_numpy().ravel()
        entities_seen_in_fit = all_ids_list[all_ids_list < fit_dim]

        entity_id_features = csr_matrix(
            (
                [1.0] * entities_seen_in_fit.shape[0],
                (entities_seen_in_fit, entities_seen_in_fit),
            ),
            shape=(matrix_height, fit_dim),
        )

        scaler_name = f"{entity}_feat_scaler"
        if getattr(self, scaler_name) is None:
            if not features_np.size:
                raise ValueError(f"features for {entity}s from log are absent")
            setattr(self, scaler_name, MinMaxScaler().fit(features_np))

        if features_np.size:
            features_np = getattr(self, scaler_name).transform(features_np)
            sparse_features = csr_matrix(
                (
                    features_np.ravel(),
                    (
                        np.repeat(entities_ids, number_of_features),
                        np.tile(
                            np.arange(number_of_features),
                            entities_ids.shape[0],
                        ),
                    ),
                ),
                shape=(matrix_height, number_of_features),
            )

        else:
            sparse_features = csr_matrix((matrix_height, number_of_features))

        concat_features = hstack([entity_id_features, sparse_features])
        concat_features_sum = concat_features.sum(axis=1).A.ravel()
        mask = concat_features_sum != 0.0
        concat_features_sum[mask] = 1.0 / concat_features_sum[mask]
        return diags(concat_features_sum, format="csr") @ concat_features

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.user_feat_scaler = None
        self.item_feat_scaler = None

        interactions_matrix = CSRConverter(
            first_dim_column="user_idx",
            second_dim_column="item_idx",
            data_column="relevance",
            row_count=self._user_dim,
            column_count=self._item_dim
        ).transform(log)
        csr_item_features = self._feature_table_to_csr(
            log.select("item_idx").distinct(), item_features
        )
        csr_user_features = self._feature_table_to_csr(
            log.select("user_idx").distinct(), user_features
        )

        if user_features is not None:
            self.can_predict_cold_users = True
        if item_features is not None:
            self.can_predict_cold_items = True

        self.model = LightFM(
            loss=self.loss,
            no_components=self.no_components,
            random_state=self.random_state,
        ).fit(
            interactions=interactions_matrix,
            epochs=self.epochs,
            num_threads=self.num_threads,
            item_features=csr_item_features,
            user_features=csr_user_features,
        )

    def _predict_selected_pairs(
        self,
        pairs: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ):
        def predict_by_user(pandas_df: pd.DataFrame) -> pd.DataFrame:
            pandas_df["relevance"] = model.predict(
                user_ids=pandas_df["user_idx"].to_numpy(),
                item_ids=pandas_df["item_idx"].to_numpy(),
                item_features=csr_item_features,
                user_features=csr_user_features,
            )
            return pandas_df

        model = self.model

        if self.can_predict_cold_users and user_features is None:
            raise ValueError("User features are missing for predict")
        if self.can_predict_cold_items and item_features is None:
            raise ValueError("Item features are missing for predict")

        csr_item_features = self._feature_table_to_csr(
            pairs.select("item_idx").distinct(), item_features
        )
        csr_user_features = self._feature_table_to_csr(
            pairs.select("user_idx").distinct(), user_features
        )

        return pairs.groupby("user_idx").applyInPandas(
            predict_by_user, REC_SCHEMA
        )

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        return self._predict_selected_pairs(
            users.crossJoin(items), user_features, item_features
        )

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        return self._predict_selected_pairs(
            pairs, user_features, item_features
        )

    def _get_features(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Tuple[Optional[DataFrame], Optional[int]]:
        """
        Get features from LightFM.
        LightFM has methods get_item_representations/get_user_representations,
        which accept object matrix and return features.

        :param ids: id item_idx/user_idx to get features for
        :param features: features for item_idx/user_idx
        :return: spark-dataframe with biases and vectors for users/items and vector size
        """
        entity = "item" if "item_idx" in ids.columns else "user"
        ids_list = ids.toPandas()[f"{entity}_idx"]

        # models without features use sparse matrix
        if features is None:
            matrix_width = getattr(self, f"fit_{entity}s").count()
            warm_ids = ids_list[ids_list < matrix_width]
            sparse_features = csr_matrix(
                (
                    [1] * warm_ids.shape[0],
                    (warm_ids, warm_ids),
                ),
                shape=(ids_list.max() + 1, matrix_width),
            )
        else:
            sparse_features = self._feature_table_to_csr(ids, features)

        biases, vectors = getattr(self.model, f"get_{entity}_representations")(
            sparse_features
        )

        embed_list = list(
            zip(
                ids_list,
                biases[ids_list].tolist(),
                vectors[ids_list].tolist(),
            )
        )
        lightfm_factors = State().session.createDataFrame(
            embed_list,
            schema=[
                f"{entity}_idx",
                f"{entity}_bias",
                f"{entity}_factors",
            ],
        )
        return lightfm_factors, self.model.no_components
