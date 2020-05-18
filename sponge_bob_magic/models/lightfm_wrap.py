"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from lightfm import LightFM
from pyspark.sql import DataFrame
from pyspark.sql.functions import PandasUDFType, pandas_udf
from pyspark.sql.types import IntegerType
from scipy.sparse import csr_matrix, hstack, identity

from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.session_handler import State
from sponge_bob_magic.utils import to_csr


class LightFMWrap(Recommender):
    """ Обёртка вокруг стандартной реализации LightFM. """

    epochs: int = 10
    num_threads: int = os.cpu_count()

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

    def get_params(self) -> Dict[str, object]:
        return {
            "no_components": self.no_components,
            "loss": self.loss,
            "random_state": self.random_state,
        }

    def _feature_table_to_csr(self, feature_table: DataFrame) -> csr_matrix:
        """
        преобразоавть свойства пользователей или объектов в разреженную матрицу

        :param feature_table: таблица с колонкой ``user_idx`` или ``item_idx``,
            все остальные колонки которой считаются значениями свойства пользователя или объекта соответстввенно
        :returns: матрица, в которой строки --- пользователи или объекты, столбцы --- их свойства
        """
        all_features = (
            State()
            .session.createDataFrame(
                range(len(self.item_indexer.labels)), schema=IntegerType()
            )
            .toDF("item_idx")
            .join(feature_table, on="item_idx", how="left")
            .fillna(0.0)
            .sort("item_idx")
            .drop("item_idx")
        )
        return hstack(
            [
                identity(self.items_count),
                csr_matrix(all_features.toPandas().to_numpy()),
            ]
        )

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.logger.debug("Построение модели LightFM")
        interactions_matrix = to_csr(log, self.users_count, self.items_count)
        csr_item_features = (
            self._feature_table_to_csr(item_features)
            if item_features is not None
            else None
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
        @pandas_udf(
            "user_idx int, item_idx int, relevance double",
            PandasUDFType.GROUPED_MAP,
        )
        def predict_by_user(pandas_df: pd.DataFrame) -> pd.DataFrame:
            pandas_df["relevance"] = model.predict(
                user_ids=pandas_df["user_idx"].to_numpy(),
                item_ids=pandas_df["item_idx"].to_numpy(),
                item_features=csr_item_features,
            )
            return pandas_df

        model = self.model
        csr_item_features = (
            self._feature_table_to_csr(item_features)
            if item_features is not None
            else None
        )
        return (
            users.crossJoin(items).groupby("user_idx").apply(predict_by_user)
        )
