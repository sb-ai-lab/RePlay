from typing import Optional, Tuple, Dict, Any

import pyspark.sql.functions as sf

from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType

from replay.models.extensions.ann.ann_mixin import ANNMixin
from replay.models.extensions.ann.index_builders.base_index_builder import IndexBuilder
from replay.models.base_rec import Recommender, ItemVectorModel
from replay.models.extensions.spark_custom_models.als_extension import ALS, ALSModel
from replay.utils.spark_utils import list_to_vector_udf


# pylint: disable=too-many-instance-attributes, too-many-ancestors
class ALSWrap(Recommender, ItemVectorModel, ANNMixin):
    """Wrapper for `Spark ALS
    <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS>`_.
    """

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        self.index_builder.index_params.dim = self.rank
        return {
            "features_col": "user_factors",
        }

    def _get_vectors_to_infer_ann_inner(self, log: DataFrame, users: DataFrame) -> DataFrame:
        user_vectors, _ = self.get_features(users)
        return user_vectors

    def _get_ann_build_params(self, log: DataFrame):
        self.index_builder.index_params.dim = self.rank
        self.index_builder.index_params.max_elements = log.select("item_idx").distinct().count()
        return {
            "features_col": "item_factors",
            "ids_col": "item_idx",
        }

    def _get_vectors_to_build_ann(self, log: DataFrame) -> DataFrame:
        item_vectors, _ = self.get_features(
            log.select("item_idx").distinct()
        )
        return item_vectors

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
        index_builder: Optional[IndexBuilder] = None,
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
        if isinstance(index_builder, (IndexBuilder, type(None))):
            self.index_builder = index_builder
        elif isinstance(index_builder, dict):
            self.init_builder_from_dict(index_builder)
        self.num_elements = None

    @property
    def _init_args(self):
        return {
            "rank": self.rank,
            "implicit_prefs": self.implicit_prefs,
            "seed": self._seed,
            "index_builder": self.index_builder.init_meta_as_dict() if self.index_builder else None,
        }

    def _save_model(self, path: str):
        self.model.write().overwrite().save(path)

        if self._use_ann:
            self._save_index(path)

    def _load_model(self, path: str):
        self.model = ALSModel.load(path)
        self.model.itemFactors.cache()
        self.model.userFactors.cache()

        if self._use_ann:
            self._load_index(path)

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        if self._num_item_blocks is None:
            self._num_item_blocks = log.rdd.getNumPartitions()
        if self._num_user_blocks is None:
            self._num_user_blocks = log.rdd.getNumPartitions()

        self.model = ALS(
            rank=self.rank,
            numItemBlocks=self._num_item_blocks,
            numUserBlocks=self._num_user_blocks,
            userCol="user_idx",
            itemCol="item_idx",
            ratingCol="relevance",
            implicitPrefs=self.implicit_prefs,
            seed=self._seed,
            coldStartStrategy="drop",
        ).fit(log)
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
        log: Optional[DataFrame],
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:

        max_seen = 0
        if filter_seen_items and log is not None:
            max_seen_in_log = (
                log.join(users, on="user_idx")
                .groupBy("user_idx")
                .agg(sf.count("user_idx").alias("num_seen"))
                .select(sf.max("num_seen"))
                .collect()[0][0]
            )
            max_seen = (
                max_seen_in_log if max_seen_in_log is not None else 0
            )

        recs_als = self.model.recommendItemsForUserItemSubset(
            users, items, k + max_seen
        )
        return (
            recs_als.withColumn(
                "recommendations", sf.explode("recommendations")
            )
            .withColumn("item_idx", sf.col("recommendations.item_idx"))
            .withColumn(
                "relevance",
                sf.col("recommendations.rating").cast(DoubleType()),
            )
            .select("user_idx", "item_idx", "relevance")
        )

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
        als_factors = getattr(self.model, f"{entity}Factors")
        als_factors = als_factors.withColumnRenamed(
            "id", f"{entity}_idx"
        ).withColumnRenamed("features", f"{entity}_factors")
        return (
            als_factors.join(ids, how="right", on=f"{entity}_idx"),
            self.model.rank,
        )

    def _get_item_vectors(self):
        return self.model.itemFactors.select(
            sf.col("id").alias("item_idx"),
            list_to_vector_udf(sf.col("features")).alias("item_vector"),
        )
