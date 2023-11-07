from os.path import join
from typing import Optional, Tuple

import pyspark.sql.functions as sf

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType
from replay.data import Dataset

from replay.models.base_rec import Recommender, ItemVectorModel
from replay.utils.spark_utils import list_to_vector_udf, save_picklable_to_parquet, load_pickled_from_parquet


# pylint: disable=too-many-instance-attributes
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
        num_query_blocks: Optional[int] = None,
    ):
        """
        :param rank: hidden dimension for the approximate matrix
        :param implicit_prefs: flag to use implicit feedback
        :param seed: random seed
        :param num_item_blocks: number of blocks the items will be partitioned into in order
            to parallelize computation.
            if None then will be init with number of partitions of log.
        :param num_query_blocks: number of blocks the queries will be partitioned into in order
            to parallelize computation.
            if None then will be init with number of partitions of log.
        """
        self.rank = rank
        self.implicit_prefs = implicit_prefs
        self._seed = seed
        self._num_item_blocks = num_item_blocks
        self._num_query_blocks = num_query_blocks

    @property
    def _init_args(self):
        return {
            "rank": self.rank,
            "implicit_prefs": self.implicit_prefs,
            "seed": self._seed,
        }

    def _save_model(self, path: str):
        self.model.write().overwrite().save(path)
        save_picklable_to_parquet(
            {
                "query_column": self.query_column,
                "item_column": self.item_column,
                "rating_column": self.rating_column,
                "timestamp_column": self.timestamp_column,
            },
            join(path, "params.dump")
        )

    def _load_model(self, path: str):
        self.model = ALSModel.load(path)
        self.model.itemFactors.cache()
        self.model.userFactors.cache()
        loaded_params = load_pickled_from_parquet(join(path, "params.dump"))
        self.query_column = loaded_params.get("query_column")
        self.item_column = loaded_params.get("item_column")
        self.rating_column = loaded_params.get("rating_column")
        self.timestamp_column = loaded_params.get("timestamp_column")

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        if self._num_item_blocks is None:
            self._num_item_blocks = dataset.interactions.rdd.getNumPartitions()
        if self._num_query_blocks is None:
            self._num_query_blocks = dataset.interactions.rdd.getNumPartitions()

        self.model = ALS(
            rank=self.rank,
            numItemBlocks=self._num_item_blocks,
            numUserBlocks=self._num_query_blocks,
            userCol=self.query_column,
            itemCol=self.item_column,
            ratingCol=self.rating_column,
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
        queries: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:

        if (items.count() == self.fit_items.count()) and (
            items.join(self.fit_items, on=self.item_column, how="inner").count()
            == self.fit_items.count()
        ):
            max_seen = 0
            if filter_seen_items and dataset.interactions is not None:
                max_seen_in_log = (
                    dataset.interactions.join(queries, on=self.query_column)
                    .groupBy(self.query_column)
                    .agg(sf.count(self.query_column).alias("num_seen"))
                    .select(sf.max("num_seen"))
                    .collect()[0][0]
                )
                max_seen = max_seen_in_log if max_seen_in_log is not None else 0

            recs_als = self.model.recommendForUserSubset(queries, k + max_seen)
            return (
                recs_als.withColumn(
                    "recommendations", sf.explode("recommendations")
                )
                .withColumn(self.item_column, sf.col(f"recommendations.{self.item_column}"))
                .withColumn(
                    self.rating_column,
                    sf.col("recommendations.rating").cast(DoubleType()),
                )
                .select(self.query_column, self.item_column, self.rating_column)
            )

        return self._predict_pairs(
            pairs=queries.crossJoin(items).withColumn(self.rating_column, sf.lit(1)),
            dataset=dataset,
        )

    def _predict_pairs(
        self,
        pairs: DataFrame,
        dataset: Optional[Dataset] = None,
    ) -> DataFrame:
        return (
            self.model.transform(pairs)
            .withColumn(self.rating_column, sf.col("prediction").cast(DoubleType()))
            .drop("prediction")
        )

    def _get_features(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Tuple[Optional[DataFrame], Optional[int]]:
        entity = "user" if self.query_column in ids.columns else "item"
        entity_col = self.query_column if self.query_column in ids.columns else self.item_column

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
            sf.col("id").alias(self.item_column),
            list_to_vector_udf(sf.col("features")).alias("item_vector"),
        )
