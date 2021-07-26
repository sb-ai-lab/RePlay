import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyspark.sql.functions as sf

from lightfm import LightFM
from pyspark.sql import DataFrame
from scipy.sparse import csr_matrix, hstack, diags
from sklearn.preprocessing import MinMaxScaler

from replay.constants import IDX_REC_SCHEMA
from replay.models.base_rec import HybridRecommender
from replay.utils import to_csr, check_numeric
from replay.session_handler import State


# pylint: disable=too-many-locals, too-many-instance-attributes
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
        loss: str = "warp",
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
        # определяют количество столбцов единичной матрицы id пользователей и объектов при построении матрицы признаков
        self.num_of_warm_items = 0
        self.num_of_warm_users = 0

    def _feature_table_to_csr(
        self,
        log_ids_list: DataFrame,
        feature_table: Optional[DataFrame] = None,
    ) -> Optional[csr_matrix]:
        """
        Преобразует признаки пользователей или объектов в разреженную матрицу
        Матрица состоит из двух частей:
        1) Левая часть соответствует ohe-hot encoding id пользователей и объектов.
        Размеры матрицы: количество пользователей/объектов * количество пользователей/объектов в fit.
        Новым пользователям/объектам соответствуют пустые строки.
        2) Правая часть содержит числовые признаки пользователей / объектов, переданные в feature_table.
        К признакам применяется MinMaxScaler по столбцам, затем элементы делятся на сумму по строке
        (сумма значений признаков по id равна 1).

        :param feature_table: таблица с колонкой ``user_idx`` или ``item_idx``,
            все остальные колонки которой считаются значениями свойства пользователя или объекта соответственно
        :param log_ids_list: таблица с колонкой ``user_idx`` или ``item_idx``, содержащая список уникальных id,
            присутствующих в логе.
        :returns: матрица, в которой строки --- пользователи или объекты, столбцы --- их свойства
        """

        if feature_table is None:
            return None

        check_numeric(feature_table)
        log_ids_list = log_ids_list.distinct()
        entity = "item" if "item_idx" in feature_table.columns else "user"
        idx_col_name = "{}_idx".format(entity)

        # оставляем признаки только для id из лога
        feature_table = feature_table.join(
            log_ids_list, on=idx_col_name, how="inner"
        )

        # определяем количество столбцов матрицы признаков
        num_entities_in_fit = getattr(self, "num_of_warm_{}s".format(entity))
        matrix_height = max(
            num_entities_in_fit,
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
                # первый столбец с id, следующие - упорядоченные признаки
                *(
                    # упорядоченные столбцы исходного датафрейма за исключением id
                    sorted(
                        list(
                            set(feature_table.columns).difference(
                                {idx_col_name}
                            )
                        )
                    )
                )
            )
            .toPandas()
            .to_numpy()
        )
        entities_ids = features_np[:, 0]
        features_np = features_np[:, 1:]
        number_of_features = features_np.shape[1]

        all_ids_list = log_ids_list.toPandas().to_numpy().ravel()
        # новым пользователям/объектам соответствуют пустые строки в столбцах, содержащих признаки id
        entities_seen_in_fit = all_ids_list[all_ids_list < num_entities_in_fit]

        # признаки, соответствующие id
        entity_id_features = csr_matrix(
            (
                [1] * entities_seen_in_fit.shape[0],
                (entities_seen_in_fit, entities_seen_in_fit),
            ),
            shape=(matrix_height, num_entities_in_fit),
        )

        # признаки из датасета
        # обучение scaler в fit
        scaler_name = "{}_feat_scaler".format(entity)
        if getattr(self, scaler_name) is None:
            if not features_np.size:
                raise ValueError(
                    "В {0}_features отсутствуют признаки для {0}s из лога".format(
                        entity
                    )
                )
            setattr(self, scaler_name, MinMaxScaler().fit(features_np))

        # применение scaler
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
        # сумма весов признаков по объекту равна 1
        concat_features_sum = diags(
            np.where(
                concat_features.sum(axis=1).A.ravel() == 0,
                0,
                1 / concat_features.sum(axis=1).A.ravel(),
            ),
            format="csr",
        )
        return concat_features_sum @ concat_features

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.user_feat_scaler = None
        self.item_feat_scaler = None

        self.num_of_warm_items = len(self.item_indexer.labels)
        self.num_of_warm_users = len(self.user_indexer.labels)

        interactions_matrix = to_csr(log, self.users_count, self.items_count)
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
            raise ValueError(
                "При обучении использовались признаки пользователей, передайте признаки для predict"
            )
        if self.can_predict_cold_items and item_features is None:
            raise ValueError(
                "При обучении использовались признаки объектов, передайте признаки для predict"
            )

        csr_item_features = self._feature_table_to_csr(
            pairs.select("item_idx").distinct(), item_features
        )
        csr_user_features = self._feature_table_to_csr(
            pairs.select("user_idx").distinct(), user_features
        )

        return pairs.groupby("user_idx").applyInPandas(
            predict_by_user, IDX_REC_SCHEMA
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
    ) -> Optional[Tuple[DataFrame, int]]:
        """
        Получение векторов пользователей и объектов из модели LightFM.
        У LightFM есть методы get_item_representations/get_user_representations,
        которые принимают матрицу признаков пользователей/объектов и возвращают вектора для выбранных пользователей.

        Внутри методов LightFM выполняется умножение переданной матрицы на матрицу,
        содержащую вектора каждого из признаков.

        :param ids: id пользователей/объектов, для которых нужно получить вектора,
            spark-dataframe с колонкой item_idx/user_idx
        :param features: spark-dataframe с колонкой item_idx/user_idx
            и колонками с признаками пользователей/объектов
        :return: spark-dataframe с bias и векторами пользователей/объектов
        """
        entity = "item" if "item_idx" in ids.columns else "user"
        ids_list = ids.toPandas()["{}_idx".format(entity)]

        # для моделей, не использующих признаки, строится разреженная матрица id пользователей/объектов
        if features is None:
            matrix_width = getattr(self, "num_of_warm_{}s".format(entity))
            warm_ids = ids_list[ids_list < matrix_width]
            sparse_features = csr_matrix(
                ([1] * warm_ids.shape[0], (warm_ids, warm_ids),),
                shape=(ids_list.max() + 1, matrix_width),
            )
        else:
            # для моделей, использующих признаки, строится полная матрица признаков
            sparse_features = self._feature_table_to_csr(ids, features)

        biases, vectors = getattr(
            self.model, "get_{}_representations".format(entity)
        )(sparse_features)

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
                "{}_idx".format(entity),
                "{}_bias".format(entity),
                "{}_factors".format(entity),
            ],
        )
        return lightfm_factors, self.model.no_components
