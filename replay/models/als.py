from typing import Optional, Tuple

import pyspark.sql.functions as sf

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType
from replay.data import Dataset

from replay.models.base_rec import Recommender, ItemVectorModel
from replay.utils.spark_utils import list_to_vector_udf


class ALSWrap(Recommender, ItemVectorModel):
    """Wrapper for `Spark ALS
    <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS>`_.
    """

    _seed: Optional[int] = None
    _search_space = {
        "rank": {"type": "loguniform_int", "args": [8, 256]},
    }

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        rank: int = 10,
        implicit_prefs: bool = True,
        seed: Optional[int] = None,
        num_item_blocks: Optional[int] = None,
        num_user_blocks: Optional[int] = None,
    ):
        """
        :param rank: hidden dimension for the approximate matrix
        :param implicit_prefs: flag to use implicit feedback
        :param seed: random seed
        :param num_item_blocks: number of blocks the items will be partitioned into in order
            to parallelize computation.
            if None then will be init with number of partitions of log.
        :param num_user_blocks: number of blocks the users will be partitioned into in order
            to parallelize computation.
            if None then will be init with number of partitions of log.
        """
        self.rank = rank
        self.implicit_prefs = implicit_prefs
        self._seed = seed
        self._num_item_blocks = num_item_blocks
        self._num_user_blocks = num_user_blocks

    @property
    def _init_args(self):
        return {
            "rank": self.rank,
            "implicit_prefs": self.implicit_prefs,
            "seed": self._seed,
        }

    def _save_model(self, path: str):
        self.model.write().overwrite().save(path)

    def _load_model(self, path: str):
        self.model = ALSModel.load(path)
        self.model.itemFactors.cache()
        self.model.userFactors.cache()

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        if self._num_item_blocks is None:
            self._num_item_blocks = dataset.interactions.rdd.getNumPartitions()
        if self._num_user_blocks is None:
            self._num_user_blocks = dataset.interactions.rdd.getNumPartitions()

        self.model = ALS(
            rank=self.rank,
            numItemBlocks=self._num_item_blocks,
            numUserBlocks=self._num_user_blocks,
            userCol=self.query_col,
            itemCol=self.item_col,
            ratingCol=self.rating_col,
            implicitPrefs=self.implicit_prefs,
            seed=self._seed,
            coldStartStrategy="drop",
        ).fit(dataset.interactions)
        self.model.itemFactors.cache()
        self.model.userFactors.cache()
        self.model.itemFactors.count()
        self.model.userFactors.count()

    def _clear_cache(self):
        if hasattr(self, "model"):
            self.model.itemFactors.unpersist()
            self.model.userFactors.unpersist()

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        dataset: Optional[Dataset],
        k: int,
        users: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:

        if (items.count() == self.fit_items.count()) and (
            items.join(self.fit_items, on=self.item_col, how="inner").count()
            == self.fit_items.count()
        ):
            max_seen = 0
            if filter_seen_items and dataset.interactions is not None:
                max_seen_in_log = (
                    dataset.interactions.join(users, on=self.query_col)
                    .groupBy(self.query_col)
                    .agg(sf.count(self.query_col).alias("num_seen"))
                    .select(sf.max("num_seen"))
                    .collect()[0][0]
                )
                max_seen = max_seen_in_log if max_seen_in_log is not None else 0

            recs_als = self.model.recommendForUserSubset(users, k + max_seen)
            return (
                recs_als.withColumn(
                    "recommendations", sf.explode("recommendations")
                )
                .withColumn(self.item_col, sf.col(f"recommendations.{self.item_col}"))
                .withColumn(
                    self.rating_col,
                    sf.col(f"recommendations.{self.rating_col}").cast(DoubleType()),
                )
                .select(self.query_col, self.item_col, self.rating_col)
            )

        return self._predict_pairs(
            pairs=users.crossJoin(items).withColumn(self.rating_col, sf.lit(1)),
            dataset=dataset,
        )

    def _predict_pairs(
        self,
        pairs: DataFrame,
        dataset: Optional[Dataset] = None,
    ) -> DataFrame:
        return (
            self.model.transform(pairs)
            .withColumn(self.rating_col, sf.col("prediction").cast(DoubleType()))
            .drop("prediction")
        )

    def _get_features(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Tuple[Optional[DataFrame], Optional[int]]:
        entity = "user" if self.query_col in ids.columns else "item"
        entity_col = self.query_col if self.query_col in ids.columns else self.item_col

        als_factors = getattr(self.model, f"{entity}Factors")
        als_factors = als_factors.withColumnRenamed(
            "id", entity_col
        ).withColumnRenamed("features", f"{entity}_factors")
        return (
            als_factors.join(ids, how="right", on=entity_col),
            self.model.rank,
        )

    def _get_item_vectors(self):
        return self.model.itemFactors.select(
            sf.col("id").alias(self.item_col),
            list_to_vector_udf(sf.col("features")).alias("item_vector"),
        )
