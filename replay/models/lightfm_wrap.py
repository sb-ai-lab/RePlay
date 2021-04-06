import os
from typing import Optional

import numpy as np
import pandas as pd
from lightfm import LightFM
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType
from scipy.sparse import csr_matrix, hstack, identity, diags
from sklearn.preprocessing import MinMaxScaler

from replay.models.base_rec import HybridRecommender
from replay.session_handler import State
from replay.utils import to_csr, check_numeric
from replay.constants import IDX_REC_SCHEMA


class LightFMWrap(HybridRecommender):
    """ Обёртка вокруг стандартной реализации LightFM. """

    epochs: int = 10
    _search_space = {
        "loss": {
            "type": "categorical",
            "args": ["logistic", "bpr", "warp", "warp-kos"],
        },
        "no_components": {"type": "loguniform_int", "args": [8, 512]},
    }

    def __init__(
        self,
        no_components: int = 128,
        loss: str = "bpr",
        random_state: Optional[int] = None,
    ):
        np.random.seed(42)
        self.no_components = no_components
        self.loss = loss
        self.random_state = random_state
        cpu_count = os.cpu_count()
        self.num_threads = cpu_count if cpu_count is not None else 1
        self.user_feat_scaler = None
        self.item_feat_scaler = None

    def _feature_table_to_csr(
        self, feature_table: Optional[DataFrame], is_item_features=True
    ) -> Optional[csr_matrix]:
        """
        Преобразует признаки пользователей или объектов в разреженную матрицу

        :param feature_table: таблица с колонкой ``user_idx`` или ``item_idx``,
            все остальные колонки которой считаются значениями свойства пользователя или объекта соответственно
        :returns: матрица, в которой строки --- пользователи или объекты, столбцы --- их свойства
        """

        if feature_table is None:
            return None

        check_numeric(feature_table)

        if is_item_features:
            df_len, idx_col_name, attr_ = (
                len(self.item_indexer.labels),
                "item_idx",
                "item_feat_scaler",
            )
        else:
            df_len, idx_col_name, attr_ = (
                len(self.user_indexer.labels),
                "user_idx",
                "user_feat_scaler",
            )

        all_features = (
            State()
            .session.createDataFrame(range(df_len), schema=IntegerType())
            .toDF(idx_col_name)
            .join(feature_table, on=idx_col_name, how="left")
            .fillna(0.0)
            .sort(idx_col_name)
            .drop(idx_col_name)
        )

        all_features_np = all_features.toPandas().to_numpy()

        if getattr(self, attr_) is None:
            setattr(self, attr_, MinMaxScaler().fit(all_features_np))
        all_features_np = getattr(self, attr_).transform(all_features_np)

        features_with_identity = hstack(
            [identity(df_len), csr_matrix(all_features_np)]
        )

        # сумма весов признаков по объекту равна 1
        features_with_identity_sum = diags(
            1 / features_with_identity.sum(axis=1).A.ravel(), format="csr"
        )
        return features_with_identity_sum @ features_with_identity

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.user_feat_scaler = None
        self.item_feat_scaler = None

        interactions_matrix = to_csr(log, self.users_count, self.items_count)
        csr_item_features = self._feature_table_to_csr(item_features)
        csr_user_features = self._feature_table_to_csr(
            user_features, is_item_features=False
        )

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
        def predict_by_user(pandas_df: pd.DataFrame) -> pd.DataFrame:
            pandas_df["relevance"] = model.predict(
                user_ids=pandas_df["user_idx"].to_numpy(),
                item_ids=pandas_df["item_idx"].to_numpy(),
                item_features=csr_item_features,
                user_features=csr_user_features,
            )
            return pandas_df

        model = self.model

        csr_item_features = self._feature_table_to_csr(item_features)
        csr_user_features = self._feature_table_to_csr(
            user_features, is_item_features=False
        )

        return (
            users.crossJoin(items)
            .groupby("user_idx")
            .applyInPandas(predict_by_user, IDX_REC_SCHEMA)
        )
