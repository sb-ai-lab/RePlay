"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.window import Window

from replay.models.base_rec import Recommender


class KNN(Recommender):
    """ Item-based KNN на сглаженной косинусной мере схожести. """

    all_items: Optional[DataFrame]
    dot_products: Optional[DataFrame]
    item_norms: Optional[DataFrame]
    similarity: Optional[DataFrame]
    _search_space = {
        "num_neighbours": {"type": "int", "args": [5, 100]},
        "shrink": {"type": "discrete_uniform", "args": [0, 50, 10]},
    }

    def __init__(self, num_neighbours: int = 10, shrink: float = 0.0):
        """
        :param num_neighbours:  ограничение на количество рассматриваемых соседей
        :param shrink: добавляется в знаменатель при подсчете сходства айтемов
        """
        self.shrink: float = shrink
        self.num_neighbours: int = num_neighbours

    def _get_similarity_matrix(
        self, items: DataFrame, dot_products: DataFrame, item_norms: DataFrame
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
            items.withColumnRenamed("item_idx", "item_id_one")
            .join(
                items.withColumnRenamed("item_idx", "item_id_two"),
                how="inner",
                on=sf.col("item_id_one") > sf.col("item_id_two"),
            )
            .join(dot_products, how="inner", on=["item_id_one", "item_id_two"])
            .join(
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
                1
                - sf.col("dot_product")
                / (sf.col("norm1") * sf.col("norm2") + self.shrink),
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
            similarity_matrix.union(
                similarity_matrix.select(
                    sf.col("item_id_two").alias("item_id_one"),
                    sf.col("item_id_one").alias("item_id_two"),
                    sf.col("similarity"),
                )
            )
            .withColumn(
                "similarity_order",
                sf.row_number().over(
                    Window.partitionBy("item_id_one").orderBy("similarity")
                ),
            )
            .filter(sf.col("similarity_order") <= self.num_neighbours)
            .drop("similarity_order")
            .cache()
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
            .groupby("item_id_one", "item_id_two")
            .agg(sf.count("user_idx").alias("dot_product"))
            .cache()
        )
        item_norms = (
            log.select("user_idx", "item_idx")
            .groupby("item_idx")
            .agg(sf.count("user_idx").alias("square_norm"))
            .select(sf.col("item_idx"), sf.sqrt("square_norm").alias("norm"))
            .cache()
        )
        all_items = log.select("item_idx").distinct().cache()

        similarity_matrix = self._get_similarity_matrix(
            all_items, dot_products, item_norms
        ).cache()

        self.similarity = self._get_k_most_similar(similarity_matrix).cache()

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
        recs = (
            log.join(users, how="inner", on="user_idx")
            .join(
                self.similarity,
                how="left",
                on=sf.col("item_idx") == sf.col("item_id_one"),
            )
            .groupby("user_idx", "item_id_two")
            .agg(sf.sum("similarity").alias("relevance"))
            .withColumnRenamed("item_id_two", "item_idx")
            .cache()
        )

        return recs
