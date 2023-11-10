from typing import Optional, Dict, Any

from pyspark.ml.feature import Word2Vec
from pyspark.ml.functions import vector_to_array
from pyspark.ml.stat import Summarizer
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from replay.models.extensions.ann.ann_mixin import ANNMixin
from replay.models.extensions.ann.index_builders.base_index_builder import IndexBuilder
from replay.models.base_rec import Recommender, ItemVectorModel
from replay.utils.spark_utils import (
    vector_dot,
    multiply_scala_udf,
    join_with_col_renaming,
)
from replay.data import Dataset


# pylint: disable=too-many-instance-attributes, too-many-ancestors
class Word2VecRec(Recommender, ItemVectorModel, ANNMixin):
    """
    Trains word2vec model where items are treated as words and queries as sentences.
    """

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        self.index_builder.index_params.dim = self.rank
        return {
            "features_col": "query_vector",
        }

    def _get_vectors_to_infer_ann_inner(self, interactions: DataFrame, queries: DataFrame) -> DataFrame:
        query_vectors = self._get_query_vectors(queries, interactions)
        # converts to pandas_udf compatible format
        query_vectors = query_vectors.select(
            self.query_column, vector_to_array("query_vector").alias("query_vector")
        )
        return query_vectors

    def _get_ann_build_params(self, interactions: DataFrame) -> Dict[str, Any]:
        self.index_builder.index_params.dim = self.rank
        self.index_builder.index_params.max_elements = interactions.select(self.item_column).distinct().count()
        self.logger.debug("index 'num_elements' = %s", self.num_elements)
        return {
            "features_col": "item_vector",
            "ids_col": self.item_column
        }

    def _get_vectors_to_build_ann(self, interactions: DataFrame) -> DataFrame:
        item_vectors = self._get_item_vectors()
        item_vectors = (
            item_vectors
            .select(
                self.item_column,
                vector_to_array("item_vector").alias("item_vector")
            )
        )
        return item_vectors

    idf: DataFrame
    vectors: DataFrame

    can_predict_cold_queries = True
    _search_space = {
        "rank": {"type": "int", "args": [50, 300]},
        "window_size": {"type": "int", "args": [1, 100]},
        "use_idf": {"type": "categorical", "args": [True, False]},
    }

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        rank: int = 100,
        min_count: int = 5,
        step_size: int = 0.025,
        max_iter: int = 1,
        window_size: int = 1,
        use_idf: bool = False,
        seed: Optional[int] = None,
        num_partitions: Optional[int] = None,
        index_builder: Optional[IndexBuilder] = None,
    ):
        """
        :param rank: embedding size
        :param min_count: the minimum number of times a token must
            appear to be included in the word2vec model's vocabulary
        :param step_size: step size to be used for each iteration of optimization
        :param max_iter: max number of iterations
        :param window_size: window size
        :param use_idf: flag to use inverse document frequency
        :param seed: random seed
        :param index_builder: `IndexBuilder` instance that adds ANN functionality.
            If not set, then ann will not be used.
        """

        self.rank = rank
        self.window_size = window_size
        self.use_idf = use_idf
        self.min_count = min_count
        self.step_size = step_size
        self.max_iter = max_iter
        self._seed = seed
        self._num_partitions = num_partitions
        if isinstance(index_builder, (IndexBuilder, type(None))):
            self.index_builder = index_builder
        elif isinstance(index_builder, dict):
            self.init_builder_from_dict(index_builder)
        self.num_elements = None

    @property
    def _init_args(self):
        return {
            "rank": self.rank,
            "window_size": self.window_size,
            "use_idf": self.use_idf,
            "min_count": self.min_count,
            "step_size": self.step_size,
            "max_iter": self.max_iter,
            "seed": self._seed,
            "index_builder": self.index_builder.init_meta_as_dict() if self.index_builder else None,
        }

    def _save_model(self, path: str):
        # # create directory on shared disk or in HDFS
        # path_info = get_filesystem(path)
        # destination_filesystem, target_dir_path = fs.FileSystem.from_uri(
        #     path_info.hdfs_uri + path_info.path
        #     if path_info.filesystem == FileSystem.HDFS
        #     else path_info.path
        # )
        # destination_filesystem.create_dir(target_dir_path)
        super()._save_model(path)
        if self.index_builder:
            self._save_index(path)

    def _load_model(self, path: str):
        super()._load_model(path)
        if self.index_builder:
            self._load_index(path)

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        self.idf = (
            dataset.interactions.groupBy(self.item_column)
            .agg(sf.countDistinct(self.query_column).alias("count"))
            .withColumn(
                "idf",
                sf.log(sf.lit(self.queries_count) / sf.col("count"))
                if self.use_idf
                else sf.lit(1.0),
            )
            .select(self.item_column, "idf")
        )
        self.idf.cache().count()

        interactions_by_queries = (
            dataset.interactions.groupBy(self.query_column)
            .agg(
                sf.collect_list(sf.struct(self.timestamp_column, self.item_column)).alias(
                    "ts_item_idx"
                )
            )
            .withColumn("ts_item_idx", sf.array_sort("ts_item_idx"))
            .withColumn(
                "items",
                sf.col(f"ts_item_idx.{self.item_column}").cast(
                    st.ArrayType(st.StringType())
                ),
            )
            .drop("ts_item_idx")
        )

        self.logger.debug("Model training")

        if self._num_partitions is None:
            self._num_partitions = interactions_by_queries.rdd.getNumPartitions()

        word_2_vec = Word2Vec(
            vectorSize=self.rank,
            minCount=self.min_count,
            numPartitions=self._num_partitions,
            stepSize=self.step_size,
            maxIter=self.max_iter,
            inputCol="items",
            outputCol="w2v_vector",
            windowSize=self.window_size,
            seed=self._seed,
        )
        self.vectors = (
            word_2_vec.fit(interactions_by_queries)
            .getVectors()
            .select(sf.col("word").cast("int").alias("item"), "vector")
        )
        self.vectors.cache().count()

    def _clear_cache(self):
        if hasattr(self, "idf") and hasattr(self, "vectors"):
            self.idf.unpersist()
            self.vectors.unpersist()

    @property
    def _dataframes(self):
        return {"idf": self.idf, "vectors": self.vectors}

    def _get_query_vectors(
        self,
        queries: DataFrame,
        interactions: DataFrame,
    ) -> DataFrame:
        """
        :param queries: query ids, dataframe ``[query_id]``
        :param interactions: interaction dataframe
            ``[query_id, item_id, timestamp, rating]``
        :return: query embeddings dataframe
            ``[query_id, query_vector]``
        """
        res = join_with_col_renaming(
            interactions, queries, on_col_name=self.query_column, how="inner"
        )
        res = join_with_col_renaming(
            res, self.idf, on_col_name=self.item_column, how="inner"
        )
        res = res.join(
            self.vectors.hint("broadcast"),
            how="inner",
            on=sf.col(self.item_column) == sf.col("item"),
        ).drop("item")
        return (
            res.groupby(self.query_column)
            .agg(
                Summarizer.mean(
                    multiply_scala_udf(sf.col("idf"), sf.col("vector"))
                ).alias("query_vector")
            )
            .select(self.query_column, "query_vector")
        )

    def _predict_pairs_inner(
        self,
        pairs: DataFrame,
        dataset: Dataset,
    ) -> DataFrame:
        if dataset is None:
            raise ValueError(
                f"interactions is not provided, {self} predict requires interactions."
            )

        query_vectors = self._get_query_vectors(
            pairs.select(self.query_column).distinct(), dataset.interactions
        )
        pairs_with_vectors = join_with_col_renaming(
            pairs, query_vectors, on_col_name=self.query_column, how="inner"
        )
        pairs_with_vectors = pairs_with_vectors.join(
            self.vectors, on=sf.col(self.item_column) == sf.col("item"), how="inner"
        ).drop("item")
        return pairs_with_vectors.select(
            self.query_column,
            sf.col(self.item_column),
            (
                vector_dot(sf.col("vector"), sf.col("query_vector"))
                + sf.lit(self.rank)
            ).alias(self.rating_column),
        )

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        dataset: Dataset,
        k: int,
        queries: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        return self._predict_pairs_inner(queries.crossJoin(items), dataset)

    def _predict_pairs(
        self,
        pairs: DataFrame,
        dataset: Optional[Dataset] = None,
    ) -> DataFrame:
        return self._predict_pairs_inner(pairs, dataset)

    def _get_item_vectors(self):
        return self.vectors.withColumnRenamed(
            "vector", "item_vector"
        ).withColumnRenamed("item", self.item_column)
