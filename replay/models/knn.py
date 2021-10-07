from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.window import Window

from replay.models.base_rec import NeighbourRec


class KNN(NeighbourRec):
    """Item-based KNN with modified cosine similarity measure."""

    all_items: Optional[DataFrame]
    dot_products: Optional[DataFrame]
    item_norms: Optional[DataFrame]
    _search_space = {
        "num_neighbours": {"type": "int", "args": [1, 100]},
        "shrink": {"type": "int", "args": [0, 100]},
    }

    def __init__(self, num_neighbours: int = 10, shrink: float = 0.0):
        """
        :param num_neighbours:  number of neighbours
        :param shrink: term added to the denominator when calculating similarity
        """
        self.shrink: float = shrink
        self.num_neighbours: int = num_neighbours

    def _get_similarity_matrix(
        self, dot_products: DataFrame, item_norms: DataFrame
    ) -> DataFrame:
        """
        Get similarity matrix

        :param dot_products: dot products between items, `[item_id_one, item_id_two, dot_product]`
        :param item_norms: euclidean norms for items `[item_id, norm]`
        :return: similarity matrix `[item_id_one, item_id_two, similarity]`
        """
        return (
            dot_products.join(
                item_norms.withColumnRenamed(
                    "item_idx", "item_id1"
                ).withColumnRenamed("norm", "norm1"),
                how="inner",
                on=sf.col("item_id1") == sf.col("item_id_one"),
            )
            .join(
                item_norms.withColumnRenamed(
                    "item_idx", "item_id2"
                ).withColumnRenamed("norm", "norm2"),
                how="inner",
                on=sf.col("item_id2") == sf.col("item_id_two"),
            )
            .withColumn(
                "similarity",
                sf.col("dot_product")
                / (sf.col("norm1") * sf.col("norm2") + self.shrink),
            )
            .select("item_id_one", "item_id_two", "similarity")
        )

    def _get_k_most_similar(self, similarity_matrix: DataFrame) -> DataFrame:
        """
        Leaves only top-k neighbours for each item

        :param similarity_matrix: dataframe `[item_id_one, item_id_two, similarity]`
        :return: cropped similarity matrix
        """
        return (
            similarity_matrix.withColumn(
                "similarity_order",
                sf.row_number().over(
                    Window.partitionBy("item_id_one").orderBy(
                        sf.col("similarity").desc(),
                        sf.col("item_id_two").desc(),
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
        dot_products = (
            log.select("user_idx", "item_idx")
            .withColumnRenamed("item_idx", "item_id_one")
            .join(
                log.select("user_idx", "item_idx").withColumnRenamed(
                    "item_idx", "item_id_two"
                ),
                how="inner",
                on="user_idx",
            )
            .filter(sf.col("item_id_one") != sf.col("item_id_two"))
            .groupby("item_id_one", "item_id_two")
            .agg(sf.count("user_idx").alias("dot_product"))
        )
        item_norms = (
            log.select("user_idx", "item_idx")
            .groupby("item_idx")
            .agg(sf.count("user_idx").alias("square_norm"))
            .select(sf.col("item_idx"), sf.sqrt("square_norm").alias("norm"))
        )

        similarity_matrix = self._get_similarity_matrix(
            dot_products, item_norms
        )

        self.similarity = self._get_k_most_similar(similarity_matrix).cache()
