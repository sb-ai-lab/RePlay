from typing import Optional, Tuple

import pyspark.sql.functions as sf

from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType

from replay.models.base_rec import Recommender
from replay.utils import (
    list_to_vector_udf,
    vector_squared_distance,
    cosine_similarity,
)


class ALSWrap(Recommender):
    """Wrapper for `Spark ALS
    <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS>`_.
    """

    can_predict_item_to_item: bool = True
    _seed: Optional[int] = None
    _search_space = {
        "rank": {"type": "loguniform_int", "args": [8, 256]},
    }

    def __init__(
        self,
        rank: int = 10,
        implicit_prefs: bool = True,
        seed: Optional[int] = None,
    ):
        """
        :param rank: hidden dimension for the approximate matrix
        :param implicit_prefs: flag to use implicit feedback
        :param seed: random seed
        """
        self.rank = rank
        self.implicit_prefs = implicit_prefs
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
            implicitPrefs=self.implicit_prefs,
            seed=self._seed,
            coldStartStrategy="drop",
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
        test_data = users.crossJoin(items).withColumn("relevance", sf.lit(1))
        recs = (
            self.model.transform(test_data)
            .withColumn("relevance", sf.col("prediction").cast(DoubleType()))
            .drop("prediction")
        )
        return recs

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        return (
            self.model.transform(pairs)
            .withColumn("relevance", sf.col("prediction").cast(DoubleType()))
            .drop("prediction")
        )

    def _get_features(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Tuple[Optional[DataFrame], Optional[int]]:
        entity = "user" if "user_idx" in ids.columns else "item"
        als_factors = getattr(self.model, "{}Factors".format(entity))
        als_factors = als_factors.withColumnRenamed(
            "id", "{}_idx".format(entity)
        ).withColumnRenamed("features", "{}_factors".format(entity))
        return (
            als_factors.join(ids, how="right", on="{}_idx".format(entity)),
            self.model.rank,
        )

    def _get_nearest_items(
        self,
        items: DataFrame,
        metric: str = "squared_distance",
        items_to_consider: Optional[DataFrame] = None,
    ) -> DataFrame:

        factor = 1
        dist_function = cosine_similarity
        if metric == "squared_distance":
            dist_function = vector_squared_distance
            factor = -1
        elif metric != "cosine_similarity":
            raise NotImplementedError(
                "{} metric is not implemented".format(metric)
            )

        als_factors = self.model.itemFactors.select(
            sf.col("id").alias("item_id_one"),
            list_to_vector_udf(sf.col("features")).alias("factors_one"),
        )

        left_part = als_factors.join(
            items.select(sf.col("item_idx").alias("item_id_one")),
            on="item_id_one",
        )

        right_part = als_factors.withColumnRenamed(
            "factors_one", "factors_two"
        ).withColumnRenamed("item_id_one", "item_id_two")

        if items_to_consider is not None:
            right_part = right_part.join(
                items_to_consider.withColumnRenamed("item_idx", "item_id_two"),
                on="item_id_two",
            )

        joined_factors = left_part.join(
            right_part, on=sf.col("item_id_one") != sf.col("item_id_two")
        )

        joined_factors = joined_factors.withColumn(
            "similarity",
            factor
            * dist_function(sf.col("factors_one"), sf.col("factors_two")),
        )

        similarity_matrix = joined_factors.select(
            "item_id_one", "item_id_two", "similarity"
        )

        return similarity_matrix
