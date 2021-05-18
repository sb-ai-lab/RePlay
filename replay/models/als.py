from typing import Optional

from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit
from pyspark.sql.types import DoubleType

from replay.models.base_rec import Recommender


class ALSWrap(Recommender):
    """ Обёртка для матричной факторизации `ALS на Spark
    <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS>`_.
    """

    _seed: Optional[int] = None
    _search_space = {
        "rank": {"type": "loguniform_int", "args": [8, 256]},
    }

    def __init__(self, rank: int = 10, seed: Optional[int] = None):
        """
        Инициализирует параметры модели и сохраняет спарк-сессию.

        :param rank: матрицей какого ранга приближаем исходную
        :param seed: random seed
        """
        self.rank = rank
        self._seed = seed

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.model = ALS(
            rank=self.rank,
            userCol="user_idx",
            itemCol="item_idx",
            ratingCol="relevance",
            implicitPrefs=True,
            seed=self._seed,
        ).fit(log)
        self.model.itemFactors.cache()
        self.model.userFactors.cache()

    def _clear_cache(self):
        if hasattr(self, "model"):
            self.model.itemFactors.unpersist()
            self.model.userFactors.unpersist()

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
        test_data = users.crossJoin(items).withColumn("relevance", lit(1))
        recs = (
            self.model.transform(test_data)
            .withColumn("relevance", col("prediction").cast(DoubleType()))
            .drop("prediction")
        )
        return recs

    def _predict_pairs_wrap(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ):
        """
        :param pairs: пары пользователь-объект, для которых необходимо сделать предсказание
        :param log: лог взаимодействий пользователей и объектов,
            спарк-датафрейм с колонками ``[user_id, item_id, relevance]``.
            Может использоваться для inference.
        :return: рекомендации, спарк-датафрейм с колонками
            ``[user_id, item_id, relevance]`` для переданных пар
        """

        users_type = pairs.schema["user_id"].dataType
        items_type = pairs.schema["item_id"].dataType
        pairs = self._convert_index(pairs.select("user_id", "item_id"))

        pred = (
            self.model.transform(pairs)
            .withColumn("relevance", col("prediction").cast(DoubleType()))
            .drop("prediction")
        )

        pred = self._convert_back(pred, users_type, items_type).select(
            "user_id", "item_id", "relevance"
        )
        return pred
