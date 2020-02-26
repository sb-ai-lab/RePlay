"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
from typing import Dict, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.window import Window

from sponge_bob_magic.constants import DEFAULT_CONTEXT
from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import get_top_k_recs, write_read_dataframe


class KNNRec(Recommender):
    """ Item-based KNN на сглаженной косинусной мере схожести. """

    all_items: Optional[DataFrame]
    dot_products: Optional[DataFrame]
    item_norms: Optional[DataFrame]
    similarity: Optional[DataFrame]

    def __init__(
            self,
            num_neighbours: int = 10,
            shrink: float = 0.0):
        self.shrink: float = shrink
        self.num_neighbours: int = num_neighbours

    def get_params(self) -> Dict[str, object]:
        return {"shrink": self.shrink,
                "num_neighbours": self.num_neighbours}

    def _get_similarity_matrix(
            self,
            items: DataFrame,
            dot_products: DataFrame,
            item_norms: DataFrame
    ) -> DataFrame:
        """
        Получает верхнюю треугольную матрицу модифицированной косинусной меры
        схожести.

        :param items: объекты, между которыми нужно посчитать схожесть,
            спарк-датафрейм с колонкой `[item_id]`
        :param dot_products: скалярные произведения между объектами,
            спарк-датафрейм вида `[item_id_one, item_id_two, dot_product]`
        :param item_norms: евклидовы нормы объектов,
            спарк-датафрейм вида `[item_id, norm]`
        :return: матрица схожести,
            спарк-датафрейм вида `[item_id_one, item_id_two, similarity]`
        """
        return (
            items
            .withColumnRenamed("item_id", "item_id_one")
            .join(
                items
                .withColumnRenamed("item_id", "item_id_two"),
                how="inner",
                on=sf.col("item_id_one") > sf.col("item_id_two")
            )
            .join(
                dot_products,
                how="inner",
                on=["item_id_one", "item_id_two"]
            )
            .join(
                item_norms.alias("item1"),
                how="inner",
                on=sf.col("item1.item_id") == sf.col("item_id_one")
            )
            .join(
                item_norms.alias("item2"),
                how="inner",
                on=sf.col("item2.item_id") == sf.col("item_id_two")
            )
            .withColumn(
                "similarity",
                1 - sf.col("dot_product") /
                (sf.col("item1.norm") * sf.col("item2.norm") + self.shrink)
            )
            .select("item_id_one", "item_id_two", "similarity")
        )

    def _get_k_most_similar(self, similarity_matrix: DataFrame) -> DataFrame:
        """
        Преобразовывает матрицу схожести:
        1) делает её симметричной;
        2) отбирает только топ-k ближайших соседей.

        :param similarity_matrix: матрица схожести,
            спарк-датафрейм вида `[item_id_one, item_id_two, similarity]`
        :return: преобразованная матрица схожести такого же вида
        """
        return (
            similarity_matrix
            .union(
                similarity_matrix
                .select(
                    sf.col("item_id_two").alias("item_id_one"),
                    sf.col("item_id_one").alias("item_id_two"),
                    sf.col("similarity")
                )
            )
            .withColumn(
                "similarity_order",
                sf.row_number().over(
                    Window.partitionBy("item_id_one").orderBy("similarity")
                )
            )
            .filter(sf.col("similarity_order") <= self.num_neighbours)
            .drop("similarity_order")
            .cache()
        )

    def _pre_fit(
            self,
            log: DataFrame,
            user_features: Optional[DataFrame],
            item_features: Optional[DataFrame]) -> None:
        self.dot_products = (
            log
            .withColumnRenamed("item_id", "item_id_one")
            .join(
                log.withColumnRenamed("item_id", "item_id_two"),
                how="inner",
                on="user_id"
            )
            .groupby("item_id_one", "item_id_two")
            .agg(sf.count("user_id").alias("dot_product"))
            .cache()
        )
        self.item_norms = (
            log
            .groupby("item_id")
            .agg(sf.count("user_id").alias("square_norm"))
            .select(sf.col("item_id"), sf.sqrt("square_norm").alias("norm"))
            .cache()
        )
        self.all_items = log.select("item_id").distinct().cache()
        spark = SparkSession(self.item_norms.rdd.context)
        self.dot_products = write_read_dataframe(
            self.dot_products,
            os.path.join(spark.conf.get("spark.local.dir"),
                         "knn_dot_products.parquet")
        )
        self.item_norms = write_read_dataframe(
            self.item_norms,
            os.path.join(spark.conf.get("spark.local.dir"),
                         "knn_item_norms.parquet")
        )
        self.all_items = write_read_dataframe(
            self.all_items,
            os.path.join(spark.conf.get("spark.local.dir"),
                         "knn_all_items.parquet")
        )

    def _fit_partial(self, log: DataFrame, user_features: Optional[DataFrame],
                     item_features: Optional[DataFrame]) -> None:
        similarity_matrix = self._get_similarity_matrix(
            self.all_items, self.dot_products, self.item_norms
        ).cache()

        self.similarity = self._get_k_most_similar(similarity_matrix).cache()
        spark = SparkSession(self.similarity.rdd.context)
        self.similarity = write_read_dataframe(
            self.similarity,
            os.path.join(spark.conf.get("spark.local.dir"),
                         "knn_similarity_matrix.parquet")
        )

    def _predict(self, log: DataFrame, k: int, users: DataFrame = None, items: DataFrame = None, context: str = None,
                 user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        recs = (
            log
            .join(
                users,
                how="inner",
                on="user_id"
            )
            .join(
                self.similarity,
                how="left",
                on=sf.col("item_id") == sf.col("item_id_one")
            )
            .groupby("user_id", "item_id_two")
            .agg(sf.sum("similarity").alias("relevance"))
            .withColumnRenamed("item_id_two", "item_id")
            .withColumn("context", sf.lit(DEFAULT_CONTEXT))
            .cache()
        )

        if filter_seen_items:
            recs = self._filter_seen_recs(recs, log)

        recs = get_top_k_recs(recs, k)
        recs = recs.filter(sf.col("relevance") > 0.0)

        return recs
