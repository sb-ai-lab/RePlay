from os.path import join
from typing import Optional, Tuple

from replay.data import Dataset
from replay.models.base_rec import ItemVectorModel, Recommender
from replay.utils import PYSPARK_AVAILABLE, SparkDataFrame

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf
    from pyspark.ml.recommendation import ALS, ALSModel
    from pyspark.sql.types import DoubleType

    from replay.utils.spark_utils import list_to_vector_udf


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
            if None then will be init with number of partitions of interactions.
        :param num_query_blocks: number of blocks the queries will be partitioned into in order
            to parallelize computation.
            if None then will be init with number of partitions of interactions.
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

    def _save_model(self, path: str, additional_params: Optional[dict] = None):
        super()._save_model(path, additional_params)
        self.model.write().overwrite().save(join(path, "model"))

    def _load_model(self, path: str):
        super()._load_model(path)
        self.model = ALSModel.load(join(path, "model"))
        self.model.itemFactors.cache()
        self.model.userFactors.cache()

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
        queries: SparkDataFrame,
        items: SparkDataFrame,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:

        if (items.count() == self.fit_items.count()) and (
            items.join(self.fit_items, on=self.item_column, how="inner").count()
            == self.fit_items.count()
        ):
            max_seen = 0
            if filter_seen_items and dataset is not None:
                max_seen_in_interactions = (
                    dataset.interactions.join(queries, on=self.query_column)
                    .groupBy(self.query_column)
                    .agg(sf.count(self.query_column).alias("num_seen"))
                    .select(sf.max("num_seen"))
                    .collect()[0][0]
                )
                max_seen = max_seen_in_interactions if max_seen_in_interactions is not None else 0

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
        pairs: SparkDataFrame,
        dataset: Optional[Dataset] = None,
    ) -> SparkDataFrame:
        return (
            self.model.transform(pairs)
            .withColumn(self.rating_column, sf.col("prediction").cast(DoubleType()))
            .drop("prediction")
        )

    def _get_features(
        self, ids: SparkDataFrame, features: Optional[SparkDataFrame]
    ) -> Tuple[Optional[SparkDataFrame], Optional[int]]:
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
