"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
from typing import Dict, Optional

import pandas as pd
from lightfm import LightFM
from pyspark.sql import DataFrame
from pyspark.sql.functions import PandasUDFType, pandas_udf
from scipy.sparse import coo_matrix

from sponge_bob_magic.models.base_rec import Recommender


class LightFMWrap(Recommender):
    """ Обёртка вокруг стандартной реализации LightFM. """

    epochs: int = 10
    loss = "bpr"

    def __init__(self, **kwargs):
        self.model_params: Dict[str, object] = kwargs

    def get_params(self) -> Dict[str, object]:
        return self.model_params

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.logger.debug("Построение модели LightFM")
        pandas_log = log.select("user_idx", "item_idx", "relevance").toPandas()
        interactions_matrix = coo_matrix(
            (pandas_log.relevance, (pandas_log.user_idx, pandas_log.item_idx)),
            shape=(self.users_count, self.items_count),
        )
        self.model = LightFM(loss="bpr", **self.model_params).fit(
            interactions=interactions_matrix,
            epochs=self.epochs,
            num_threads=os.cpu_count(),
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
            )
            return pandas_df

        model = self.model
        return (
            self.user_indexer.transform(users)
            .crossJoin(self.item_indexer.transform(items))
            .groupby("user_idx")
            .apply(predict_by_user)
        )
