"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from math import log2

from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from sponge_bob_magic.metrics.base_metrics import Metric, NumType
from sponge_bob_magic.utils import get_top_k_recs


class HitRate(Metric):
    """
    Метрика HitRate@K:
    для какой доли пользователей удалось порекомендовать среди
    первых `k` хотя бы один объект из реального лога.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def __str__(self):
        return "HitRate@K"

    def __call__(
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


class NDCG(Metric):
    """
    Метрика nDCG@k:
    чем релевантнее элементы среди первых `k`, тем выше метрика.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def __str__(self):
        return "nDCG@k"

    def __call__(
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


class Precision(Metric):
    """
    Метрика Precision@k:
    точность на `k` первых элементах выдачи.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def __str__(self):
        return "Precision@k"

    def __call__(
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


class MAP(Metric):
    """
    Метрика MAP@k (mean average precision):
    средняя точность на `k` первых элементах выдачи.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def __str__(self):
        return "MAP@k"

    def __call__(
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


class Recall(Metric):
    """
    Метрика Recall@K:
    какую долю объектов из реального лога мы покажем в рекомендациях среди
    первых `k` (в среднем по пользователям).
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def __str__(self):
        return "Recall@K"

    def __call__(
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


class Surprisal(Metric):
    """
    Метрика Surprisal@k:
    среднее по пользователям,
    среднее по списку рекомендаций длины k
    значение surprisal для объекта в рекомендациях.

    surprisal(item) = -log2(prob(item)),
    prob(item) =  # users which interacted with item / # total users.

    Чем выше метрика, тем больше непопулярных объектов попадают в рекомендации.

    Если normalize=True, то метрика нормирована в отрезок [0, 1].
    Для холодных объектов количество взаимодействий считается равным 1.
    """

    def __str__(self):
        return "Surprisal@K"

    def __init__(self,
                 log: DataFrame,
                 normalize: bool = False):
        """
        Считает популярность и собственную информацию каждого объета.

        :param log: спарк-датафрейм вида
            `[user_id, item_id, timestamp, context, relevance]`;
            содержит информацию о взаимодействии пользователей с объектами
        """
        n_users = log.select("user_id").distinct().count()
        max_value = -log2(1 / n_users)
        stats = log.groupby("item_id").agg(
            sf.countDistinct("user_id").alias("count")
        )

        stats = stats.withColumn("popularity", stats["count"] / n_users)
        stats = stats.withColumn("self-information", -sf.log2("popularity"))
        stats = stats.withColumn(
            "normalized_si",
            stats["self-information"] / max_value
        )

        self.stats = stats
        self.normalize = normalize
        self.fill_value = 1.0 if normalize else max_value

    def __call__(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        metric = "normalized_si" if self.normalize else "self-information"

        self_information = self.stats.select(["item_id", metric])
        top_k_recommendations = get_top_k_recs(recommendations, k)

        recs = top_k_recommendations.join(self_information,
                                          on="item_id",
                                          how="left").fillna(self.fill_value)
        list_mean = (
            recs
            .groupby("user_id")
            .agg(sf.mean(metric).alias(metric))
        )

        global_mean = (
            list_mean
            .select(sf.mean(metric).alias("mean"))
            .collect()[0]["mean"]
        )

        return global_mean
