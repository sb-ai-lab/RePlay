"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, Optional

from pyspark.ml.recommendation import ALS as SparkALS
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit
from pyspark.sql.types import DoubleType

from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import get_top_k_recs


class ALS_wrap(Recommender):
    """ Обёртка для матричной факторизации `ALS на Spark
    <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS>`_.
    """
    _seed: Optional[int] = None

    def __init__(self, rank: int = 10, seed: Optional[int] = None):
        """
        Инициализирует параметры модели и сохраняет спарк-сессию.

        :param rank: матрицей какого ранга приближаем исходную
        :param seed: random seed
        """
        self.rank = rank
        self._seed = seed

    def get_params(self) -> Dict[str, object]:
        return {
            "rank": self.rank
        }

    def _fit(self,
             log: DataFrame,
             user_features: Optional[DataFrame] = None,
             item_features: Optional[DataFrame] = None) -> None:
        self.logger.debug("Индексирование данных")
        log_indexed = self.user_indexer.transform(log)
        log_indexed = self.item_indexer.transform(log_indexed)

        self.logger.debug("Обучение модели")
        self.model = SparkALS(
            rank=self.rank,
            userCol="user_idx",
            itemCol="item_idx",
            ratingCol="relevance",
            implicitPrefs=True,
            seed=self._seed
        ).fit(log_indexed)

    def _predict(self,
                 log: DataFrame,
                 k: int,
                 users: DataFrame = None,
                 items: DataFrame = None,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        test_data = users.crossJoin(items).withColumn("relevance", lit(1))

        if filter_seen_items:
            test_data = self._filter_seen_recs(
                test_data,
                log
            ).drop("relevance")

        log_indexed = self.user_indexer.transform(test_data)
        log_indexed = self.item_indexer.transform(log_indexed)

        recs = (
            self.model.transform(log_indexed)
            .withColumn("relevance", col("prediction").cast(DoubleType()))
            .drop("user_idx", "item_idx", "prediction")
            .cache()
        )
        recs = get_top_k_recs(recs, k)
        return recs
