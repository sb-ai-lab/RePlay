from typing import Optional, Union, Dict, Any

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.window import Window
from pyspark.sql.functions import pandas_udf
from scipy.sparse import csr_matrix

from replay.models.base_rec import NeighbourRec
from replay.optuna_objective import ItemKNNObjective
from replay.session_handler import State


class ItemKNN(NeighbourRec):
    """Item-based ItemKNN with modified cosine similarity measure."""

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        return {
            "features_col": "",
            "params": self._nmslib_hnsw_params,
            "index_type": "sparse",
        }

    def _get_vectors_to_infer_ann_inner(
        self, log: DataFrame, users: DataFrame
    ) -> DataFrame:

        user_vectors = (
            log.groupBy("user_idx").agg(
            sf.collect_list("item_idx").alias("vector_items"), sf.collect_list("relevance").alias("vector_relevances"))
        )


        return user_vectors
        # @pandas_udf
        # def get_csr_matrix(
        #     user_idx: pd.Series,
        #     item_idx: pd.Series,
        #     relevance: pd.Series,
        #     ) -> pd.DataFrame:
        #
        #     user_vectors = csr_matrix(
        #         (
        #             relevance,
        #             (user_idx, item_idx),
        #         ),
        #         shape=(self._user_dim, self._item_dim),
        #     )
        #
        #     return user_vectors
        # user_vectors = get_csr_matrix(
        #     user_vectors.select("user_idx").toPandas().values,
        #     user_vectors.select("vector_items").toPandas().values,
        #     user_vectors.select("vector_relevances").toPandas().values
        # )
        # return user_vectors

    @property
    def _use_ann(self) -> bool:
        return self._nmslib_hnsw_params is not None

    all_items: Optional[DataFrame]
    dot_products: Optional[DataFrame]
    item_norms: Optional[DataFrame]
    bm25_k1 = 1.2
    bm25_b = 0.75
    _objective = ItemKNNObjective
    _search_space = {
        "num_neighbours": {"type": "int", "args": [1, 100]},
        "shrink": {"type": "int", "args": [0, 100]},
        "weighting": {"type": "categorical", "args": [None, "tf_idf", "bm25"]}
    }

    def __init__(
        self,
        num_neighbours: int = 10,
        use_relevance: bool = False,
        shrink: float = 0.0,
        weighting: str = None,
        nmslib_hnsw_params: Optional[dict] = None,
    ):
        """
        :param num_neighbours: number of neighbours
        :param use_relevance: flag to use relevance values as is or to treat them as 1
        :param shrink: term added to the denominator when calculating similarity
        :param weighting: item reweighting type, one of [None, 'tf_idf', 'bm25']
        """
        self.shrink = shrink
        self.use_relevance = use_relevance
        self.num_neighbours = num_neighbours

        valid_weightings = self._search_space["weighting"]["args"]
        if weighting not in valid_weightings:
            raise ValueError(f"weighting must be one of {valid_weightings}")
        self.weighting = weighting
        self._nmslib_hnsw_params = nmslib_hnsw_params

    @property
    def _init_args(self):
        return {
            "shrink": self.shrink,
            "use_relevance": self.use_relevance,
            "num_neighbours": self.num_neighbours,
            "weighting": self.weighting,
            "nmslib_hnsw_params": self._nmslib_hnsw_params,
        }

    def _save_model(self, path: str):
        if self._nmslib_hnsw_params:
            self._save_nmslib_hnsw_index(path, sparse=True)

    def _load_model(self, path: str):
        if self._nmslib_hnsw_params:
            self._load_nmslib_hnsw_index(path, sparse=True)

    @staticmethod
    def _shrink(dot_products: DataFrame, shrink: float) -> DataFrame:
        return dot_products.withColumn(
            "similarity",
            sf.col("dot_product")
            / (sf.col("norm1") * sf.col("norm2") + shrink),
        ).select("item_idx_one", "item_idx_two", "similarity")

    def _get_similarity(self, log: DataFrame) -> DataFrame:
        """
        Calculate item similarities

        :param log: DataFrame with interactions, `[user_idx, item_idx, relevance]`
        :return: similarity matrix `[item_idx_one, item_idx_two, similarity]`
        """
        dot_products = self._get_products(log)
        similarity = self._shrink(dot_products, self.shrink)
        return similarity

    def _reweight_log(self, log: DataFrame):
        """
        Reweight relevance according to TD-IDF or BM25 weighting.

        :param log: DataFrame with interactions, `[user_idx, item_idx, relevance]`
        :return: log `[user_idx, item_idx, relevance]`
        """
        if self.weighting == "bm25":
            log = self._get_tf_bm25(log)

        idf = self._get_idf(log)

        log = log.join(idf, how="inner", on="user_idx").withColumn(
            "relevance",
            sf.col("relevance") * sf.col("idf"),
        )

        return log

    def _get_tf_bm25(self, log: DataFrame):
        """
        Adjust relevance by BM25 term frequency.

        :param log: DataFrame with interactions, `[user_idx, item_idx, relevance]`
        :return: log `[user_idx, item_idx, relevance]`
        """
        item_stats = log.groupBy("item_idx").agg(
            sf.count("user_idx").alias("n_users_per_item")
        )
        avgdl = item_stats.select(sf.mean("n_users_per_item")).take(1)[0][0]
        log = log.join(item_stats, how="inner", on="item_idx")

        log = (
            log.withColumn(
                "relevance",
                sf.col("relevance") * (self.bm25_k1 + 1) / (
                    sf.col("relevance") + self.bm25_k1 * (
                        1 - self.bm25_b + self.bm25_b * (
                            sf.col("n_users_per_item") / avgdl
                        )
                    )
                )
            )
            .drop("n_users_per_item")
        )

        return log

    def _get_idf(self, log: DataFrame):
        """
        Return inverse document score for log reweighting.

        :param log: DataFrame with interactions, `[user_idx, item_idx, relevance]`
        :return: idf `[idf]`
        :raises: ValueError if self.weighting not in ["tf_idf", "bm25"]
        """
        df = log.groupBy("user_idx").agg(sf.count("item_idx").alias("DF"))
        n_items = log.select("item_idx").distinct().count()

        if self.weighting == "tf_idf":
            idf = (
                df.withColumn("idf", sf.log1p(sf.lit(n_items) / sf.col("DF")))
                .drop("DF")
            )
        elif self.weighting == "bm25":
            idf = (
                df.withColumn(
                    "idf",
                    sf.log1p(
                        (sf.lit(n_items) - sf.col("DF") + 0.5)
                        / (sf.col("DF") + 0.5)
                    ),
                )
                .drop("DF")
            )
        else:
            raise ValueError("weighting must be one of ['tf_idf', 'bm25']")

        return idf

    def _get_products(self, log: DataFrame) -> DataFrame:
        """
        Calculate item dot products

        :param log: DataFrame with interactions, `[user_idx, item_idx, relevance]`
        :return: similarity matrix `[item_idx_one, item_idx_two, norm1, norm2, dot_product]`
        """
        if self.weighting:
            log = self._reweight_log(log)

        left = log.withColumnRenamed(
            "item_idx", "item_idx_one"
        ).withColumnRenamed("relevance", "rel_one")
        right = log.withColumnRenamed(
            "item_idx", "item_idx_two"
        ).withColumnRenamed("relevance", "rel_two")

        dot_products = (
            left.join(right, how="inner", on="user_idx")
            .filter(sf.col("item_idx_one") != sf.col("item_idx_two"))
            .withColumn("relevance", sf.col("rel_one") * sf.col("rel_two"))
            .groupBy("item_idx_one", "item_idx_two")
            .agg(sf.sum("relevance").alias("dot_product"))
        )

        item_norms = (
            log.withColumn("relevance", sf.col("relevance") ** 2)
            .groupBy("item_idx")
            .agg(sf.sum("relevance").alias("square_norm"))
            .select(sf.col("item_idx"), sf.sqrt("square_norm").alias("norm"))
        )
        norm1 = item_norms.withColumnRenamed(
            "item_idx", "item_id1"
        ).withColumnRenamed("norm", "norm1")
        norm2 = item_norms.withColumnRenamed(
            "item_idx", "item_id2"
        ).withColumnRenamed("norm", "norm2")

        dot_products = dot_products.join(
            norm1, how="inner", on=sf.col("item_id1") == sf.col("item_idx_one")
        )
        dot_products = dot_products.join(
            norm2, how="inner", on=sf.col("item_id2") == sf.col("item_idx_two")
        )

        return dot_products

    def _get_k_most_similar(self, similarity_matrix: DataFrame) -> DataFrame:
        """
        Leaves only top-k neighbours for each item

        :param similarity_matrix: dataframe `[item_idx_one, item_idx_two, similarity]`
        :return: cropped similarity matrix
        """
        return (
            similarity_matrix.withColumn(
                "similarity_order",
                sf.row_number().over(
                    Window.partitionBy("item_idx_one").orderBy(
                        sf.col("similarity").desc(),
                        sf.col("item_idx_two").desc(),
                    )
                ),
            )
            .filter(sf.col("similarity_order") <= self.num_neighbours)
            .drop("similarity_order")
        )

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        df = log.select("user_idx", "item_idx", "relevance")
        if not self.use_relevance:
            df = df.withColumn("relevance", sf.lit(1))

        similarity_matrix = self._get_similarity(df)
        self.similarity = self._get_k_most_similar(similarity_matrix)
        self.similarity.cache().count()

    def refit(
        self,
        log: DataFrame,
        previous_log: Optional[Union[str, DataFrame]] = None,
        merged_log_path: Optional[str] = None,
    ) -> None:
        pass

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

        return self._predict_pairs_inner(
            log=log,
            filter_df=items.withColumnRenamed("item_idx", "item_idx_filter"),
            condition=sf.col("item_idx_two") == sf.col("item_idx_filter"),
            users=users,
        )
