"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту
"""
from typing import Dict, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from sponge_bob_magic.constants import DEFAULT_CONTEXT
from sponge_bob_magic.models.base_recommender import BaseRecommender


class KNNRecommender(BaseRecommender):
    """ item-based KNN на сглаженной косинусной мере схожести """
    similarity: DataFrame

    def __init__(self, spark: SparkSession, k: int, shrink: float = 0.0):
        super().__init__(spark)
        self.shrink = shrink
        self.k = k

    def get_params(self) -> Dict[str, object]:
        """ показать параметры модели """
        return {"shrink": self.shrink, "k": self.k}

    def _get_similarity_matrix(
            self,
            items: DataFrame,
            dot_products: DataFrame,
            item_norms: DataFrame
    ) -> DataFrame:
        """
        получить верхнюю треугольную матрицу модифицированной косинусной меры
        схожести

        :param items: объекты, между которыми нужно посчитать схожесть.
        Содержит только одну колонку: item_id
        :param dot_products: скалярные произведения между объектами. Имеет вид
        (item_id_one, item_id_two, dot_product)
        :param item_norms: евклидовы нормы объектов. Имеет вид
        (item_id, norm)
        :returns: матрица схожести вида
        (item_id_one, item_id_two, similarity)
        """
        return (
            items
            .withColumnRenamed("item_id", "item_id_one")
            .join(
                items
                .withColumnRenamed("item_id", "item_id_two"),
                how="inner",
                on=F.col("item_id_one") > F.col("item_id_two")
            )
            .join(
                dot_products,
                how="inner",
                on=["item_id_one", "item_id_two"]
            )
            .join(
                item_norms.alias("item1"),
                how="inner",
                on=F.col("item1.item_id") == F.col("item_id_one")
            )
            .join(
                item_norms.alias("item2"),
                how="inner",
                on=F.col("item2.item_id") == F.col("item_id_two")
            )
            .withColumn(
                "similarity",
                F.col("dot_product") /
                (F.col("item1.norm") * F.col("item2.norm") + self.shrink)
            )
            .select("item_id_one", "item_id_two", "similarity")
        )

    def _get_k_most_similar(self, similarity_matrix: DataFrame) -> DataFrame:
        """
        преобразовать матрицу схожести:
        1) сделать её симметричной
        2) отобрать только топ k ближайших соседей

        :param similarity_matrix: матрица схожести вида
        (item_id_one, item_id_two, similarity)
        """
        return (
            similarity_matrix
            .union(
                similarity_matrix
                .select(
                    F.col("item_id_two").alias("item_id_one"),
                    F.col("item_id_one").alias("item_id_two"),
                    F.col("similarity")
                )
            )
            .withColumn(
                "similarity_order",
                F.row_number().over(
                    Window.partitionBy("item_id_one").orderBy("similarity")
                )
            )
            .filter(F.col("similarity_order") <= self.k)
            .drop("similarity_order")
            .cache()
        )

    def _fit(
            self,
            log: DataFrame,
            user_features: Optional[DataFrame],
            item_features: Optional[DataFrame],
            path: Optional[str] = None
    ) -> None:
        dot_products = (
            log
            .withColumnRenamed("item_id", "item_id_one")
            .join(
                log.withColumnRenamed("item_id", "item_id_two"),
                how="inner",
                on="user_id"
            )
            .groupby("item_id_one", "item_id_two")
            .agg(F.count("user_id").alias("dot_product"))
            .cache()
        )
        item_norms = (
            log
            .groupby("item_id")
            .agg(F.count("user_id").alias("square_norm"))
            .select(F.col("item_id"), F.sqrt("square_norm").alias("norm"))
            .cache()
        )
        items = log.select("item_id").distinct().cache()
        similarity_matrix = self._get_similarity_matrix(
            items, dot_products, item_norms
        ).cache()
        self.similarity = self._get_k_most_similar(similarity_matrix).cache()

    def _predict(
            self,
            k: int,
            users: DataFrame,
            items: DataFrame,
            context: str,
            log: DataFrame,
            user_features: Optional[DataFrame],
            item_features: Optional[DataFrame],
            to_filter_seen_items: bool = True,
            path: Optional[str] = None
    ) -> DataFrame:
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
                on=F.col("item_id") == F.col("item_id_one")
            )
            .groupby("user_id", "item_id_two")
            .agg(F.sum("similarity").alias("relevance"))
            .withColumnRenamed("item_id_two", "item_id")
            .withColumn("context", F.lit(DEFAULT_CONTEXT))
            .cache()
        )
        if to_filter_seen_items:
            recs = self._filter_seen_recs(recs, log)
        recs = self._get_top_k_recs(recs, k)
        recs = recs.filter(F.col("relevance") > 0.0)
        return recs.cache()
