"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Optional

from pyspark.ml.feature import Word2Vec
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from pyspark.ml.stat import Summarizer

from replay.models.base_rec import Recommender
from replay.utils import vector_dot, vector_mult


class Word2VecRec(Recommender):
    """
        Рекомендатель на основе word2vec, в котором items сопоставляются
        словам, а пользователи предложениям.
    """

    idf: DataFrame
    vectors: DataFrame
    _search_space = {
        "rank": {"type": "int", "args": [50, 300]},
        "window_size": {"type": "int", "args": [1, 100]},
        "use_idf": {"type": "categorical", "args": [True, False]},
    }

    def __init__(
        self,
        rank: int = 100,
        window_size: int = 1,
        use_idf: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Инициализирует параметры модели .

        :param rank: размерность вектора ембеддингов
        :param window_size: размер окна
        :param use_idf: использовать ли взвешенное суммирование векторов с
            помощью idf
        :param seed: random seed
        """

        self.rank = rank
        self.window_size = window_size
        self.use_idf = use_idf
        self._seed = seed

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.idf = (
            log.orderBy("timestamp")
            .groupBy("item_idx")
            .agg(sf.countDistinct("user_idx").alias("count"))
            .select(
                "item_idx",
                (
                    sf.log(self.users_count / sf.col("count"))
                    if self.use_idf
                    else sf.lit(1.0)
                ).alias("idf"),
            )
        ).cache()
        log_by_users = (
            log.orderBy("timestamp")
            .groupBy("user_idx")
            .agg(
                sf.collect_list("item_idx")
                .cast(st.ArrayType(st.StringType()))
                .alias("items")
            )
        )
        self.logger.debug("Обучение модели")
        word_2_vec = Word2Vec(
            vectorSize=self.rank,
            minCount=0,
            inputCol="items",
            outputCol="w2v_vector",
            windowSize=self.window_size,
            seed=self._seed,
        )
        self.vectors = (
            word_2_vec.fit(log_by_users)
            .getVectors()
            .select(sf.col("word").cast("int").alias("item"), "vector")
            .cache()
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
        idf = self.idf.join(
            items.withColumnRenamed("item_idx", "item"),
            how="inner",
            on=sf.col("item") == sf.col("item_idx"),
        ).select("item_idx", "idf")
        user_vectors = (
            users.join(log, how="inner", on="user_idx")
            .join(idf, how="inner", on="item_idx")
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
        recs = (
            user_vectors.crossJoin(self.vectors).select(
                "user_idx",
                (
                    vector_dot(sf.col("vector"), sf.col("user_vector"))
                    + sf.lit(self.rank)
                ).alias("relevance"),
                sf.col("item").alias("item_idx"),
            )
        ).cache()
        return recs
