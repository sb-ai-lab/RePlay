"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, desc, row_number


class Metrics:
    """ различные метрики качества рекомендательных систем """

    @staticmethod
    def hit_rate_at_k(
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> float:
        """
        hit-rate: для какой доли пользователей удалось порекомендовать среди
        первых `k` хотя бы один объект из реального лога

        :param recommendations: выдача рекомендательной системы вида
        (user_id, item_id, context, relevance)
        :param ground_truth: реальный лог действий пользователей
        :param k: какое максимальное количество объектов брать из топа
        рекомендованных для оценки
        """
        top_k_recommendations = recommendations.withColumn(
            "relevance_rank",
            row_number().over(
                Window.partitionBy("user_id", "context")
                .orderBy(desc("relevance"))
            )
        ).filter(col("relevance_rank") <= k).drop("relevance_rank")
        users_hit = top_k_recommendations.join(
            ground_truth,
            how="inner",
            on=["user_id", "item_id", "context"]
        ).select("user_id").distinct().count()
        users_total = (
            top_k_recommendations.select("user_id").distinct().count()
        )
        return users_hit / users_total
