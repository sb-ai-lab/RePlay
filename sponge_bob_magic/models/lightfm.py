"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, Optional

import numpy as np
from lightfm import LightFM
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit
from scipy.sparse import coo_matrix

from sponge_bob_magic.models.base_rec import Recommender


class LightFMWrap(Recommender):
    """ Обёртка вокруг стандартной реализации LightFM. """

    _seed: Optional[int] = None

    def __init__(self, rank: int = 10, seed: Optional[int] = None):
        """
        Инициализирует параметры модели и сохраняет спарк-сессию.

        :param rank: матрицей какого ранга приближаем исходную
        """
        self.rank = rank
        self._seed = seed

    def get_params(self) -> Dict[str, object]:
        return {"rank": self.rank}

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.logger.debug("Построение модели LightFM")
        log_indexed = self.user_indexer.transform(log)
        log_indexed = self.item_indexer.transform(log_indexed)
        pandas_log = log_indexed.select("user_idx", "item_idx", "relevance").toPandas()
        interactions_matrix = coo_matrix(
            (pandas_log.relevance, (pandas_log.user_idx, pandas_log.item_idx)),
            shape=(self.users_count, self.items_count),
        )
        self.model = LightFM(
            no_components=self.rank, loss="bpr", random_state=self._seed
        ).fit(interactions=interactions_matrix, epochs=10, num_threads=1)

    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame = None,
        items: DataFrame = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        test_data = users.crossJoin(items).withColumn("relevance", lit(1))
        if filter_seen_items:
            test_data = self._filter_seen_recs(test_data, log).drop("relevance")
        log_indexed = self.user_indexer.transform(test_data)
        log_indexed = self.item_indexer.transform(log_indexed)
        prediction = log_indexed.toPandas()
        prediction["relevance"] = self.model.predict(
            np.array(prediction.user_idx), np.array(prediction.item_idx)
        )
        recs = self.spark.createDataFrame(
            prediction[["user_id", "item_id", "relevance"]]
        ).cache()
        return recs
