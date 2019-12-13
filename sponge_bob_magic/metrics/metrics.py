"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from sponge_bob_magic.metrics.base_metrics import BaseMetrics, NumType
from sponge_bob_magic.utils import get_top_k_recs


class HitRateMetric(BaseMetrics):
    """
    Метрика HitRate@K:
    для какой доли пользователей удалось порекомендовать среди
    первых `k` хотя бы один объект из реального лога.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def calculate(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        top_k_recommendations = get_top_k_recs(recommendations, k)
        users_hit = top_k_recommendations.join(
            ground_truth,
            how="inner",
            on=["user_id", "item_id", "context"]
        ).select("user_id").distinct().count()
        users_total = (
            top_k_recommendations.select("user_id").distinct().count()
        )
        return users_hit / users_total


class NDCGMetric(BaseMetrics):
    """
    Метрика nDCG@k:
    чем релевантнее элементы среди первых `k`, тем выше метрика.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def calculate(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        dataframe = self._merge_prediction_and_truth(
            recommendations, ground_truth
        )
        dataframe = dataframe.map(lambda row: (row[1], row[2]))
        metrics = RankingMetrics(dataframe)
        return metrics.ndcgAt(k)


class PrecisionMetric(BaseMetrics):
    """
    Метрика Precision@k:
    точность на `k` первых элементах выдачи.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def calculate(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        dataframe = self._merge_prediction_and_truth(
            recommendations, ground_truth
        )
        dataframe = dataframe.map(lambda row: (row[1], row[2]))
        metrics = RankingMetrics(dataframe)
        return metrics.precisionAt(k)


class MAPMetric(BaseMetrics):
    """
    Метрика MAP@k (mean average precision):
    средняя точность на `k` первых элементах выдачи.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def calculate(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        dataframe = self._merge_prediction_and_truth(
            recommendations, ground_truth
        )
        dataframe = dataframe.map(lambda row: (row[1][:k], row[2][:k]))
        metrics = RankingMetrics(dataframe)
        return metrics.meanAveragePrecision


class RecallMetric(BaseMetrics):
    """
    Метрика recall@K:
    какую долю объектов из реального лога мы покажем в рекомендациях среди
    первых `k` (в среднем по пользователям).
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def calculate(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        top_k_recommendations = get_top_k_recs(recommendations, k)
        hits = top_k_recommendations.join(
            ground_truth,
            how="inner",
            on=["user_id", "item_id"]
        ).groupby("user_id").agg(sf.count("item_id").alias("hits"))
        totals = (
            ground_truth
            .groupby("user_id")
            .agg(sf.count("item_id").alias("totals"))
        )
        total_recall, total_users = (
            totals.join(hits, on=["user_id"], how="left")
            .withColumn(
                "recall",
                sf.coalesce(sf.col("hits") / sf.col("totals"), sf.lit(0))
            )
            .agg(
                sf.sum("recall").alias("total_hits"),
                sf.count("recall").alias("total_users")
            )
            .select("total_hits", "total_users")
            .head()[:2]
        )
        return total_recall / total_users
