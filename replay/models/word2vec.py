from typing import Optional

from pyspark.ml.feature import Word2Vec
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from pyspark.ml.stat import Summarizer

from replay.models.base_rec import Recommender, ItemVectorModel
from replay.utils import vector_dot, vector_mult


# pylint: disable=too-many-instance-attributes
class Word2VecRec(Recommender, ItemVectorModel):
    """
    Trains word2vec model where items ar treated as words and users as sentences.
    """

    idf: DataFrame
    vectors: DataFrame

    can_predict_cold_users = True
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
        """

        self.rank = rank
        self.window_size = window_size
        self.use_idf = use_idf
        self.min_count = min_count
        self.step_size = step_size
        self.max_iter = max_iter
        self._seed = seed

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.idf = (
            log.groupBy("item_idx")
            .agg(sf.countDistinct("user_idx").alias("count"))
            .select(
                "item_idx",
                (
                    sf.log(self.users_count / sf.col("count"))
                    if self.use_idf
                    else sf.lit(1.0)
                ).alias("idf"),
            )
        )
        self.idf.cache()

        log_by_users = (
            log.groupBy("user_idx")
            .agg(
                sf.collect_list(sf.struct("timestamp", "item_idx")).alias(
                    "ts_item_idx"
                )
            )
            .withColumn("ts_item_idx", sf.array_sort("ts_item_idx"))
            .withColumn(
                "items",
                sf.col("ts_item_idx.item_idx").cast(
                    st.ArrayType(st.StringType())
                ),
            )
            .drop("ts_item_idx")
        )

        self.logger.debug("Model training")

        word_2_vec = Word2Vec(
            vectorSize=self.rank,
            minCount=self.min_count,
            stepSize=self.step_size,
            maxIter=self.max_iter,
            inputCol="items",
            outputCol="w2v_vector",
            windowSize=self.window_size,
            seed=self._seed,
        )
        self.vectors = (
            word_2_vec.fit(log_by_users)
            .getVectors()
            .select(sf.col("word").cast("int").alias("item"), "vector")
        )
        self.vectors.cache()

    def _clear_cache(self):
        if hasattr(self, "idf") and hasattr(self, "vectors"):
            self.idf.unpersist()
            self.vectors.unpersist()

    def _get_user_vectors(
        self,
        users: DataFrame,
        log: DataFrame,
    ) -> DataFrame:
        """
        :param users: user ids, dataframe ``[user_idx]``
        :param log: interaction dataframe
            ``[user_idx, item_idx, timestamp, relevance]``
        :return: user embeddings dataframe
            ``[user_idx, user_vector]``
        """
        return (
            log.join(users, how="inner", on="user_idx")
            .join(self.idf, how="inner", on="item_idx")
            .join(
                self.vectors,
                how="inner",
                on=sf.col("item_idx") == sf.col("item"),
            )
            .groupby("user_idx")
            .agg(
                Summarizer.mean(
                    vector_mult(sf.col("idf"), sf.col("vector"))
                ).alias("user_vector")
            )
            .select("user_idx", "user_vector")
        )

    def _predict_pairs_inner(
        self,
        pairs: DataFrame,
        log: DataFrame,
    ) -> DataFrame:
        if log is None:
            raise ValueError(
                f"log is not provided, {self.__str__()} predict requires log."
            )

        user_vectors = self._get_user_vectors(
            pairs.select("user_idx").distinct(), log
        )
        pairs_with_vectors = pairs.join(
            user_vectors, on="user_idx", how="inner"
        ).join(
            self.vectors, on=sf.col("item_idx") == sf.col("item"), how="inner"
        )
        return pairs_with_vectors.select(
            "user_idx",
            sf.col("item_idx"),
            (
                vector_dot(sf.col("vector"), sf.col("user_vector"))
                + sf.lit(self.rank)
            ).alias("relevance"),
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
        return self._predict_pairs_inner(users.crossJoin(items), log)

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        return self._predict_pairs_inner(pairs, log)

    def _get_item_vectors(self):
        return self.vectors.withColumnRenamed(
            "vector", "item_vector"
        ).withColumnRenamed("item", "item_idx")
