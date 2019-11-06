"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf


class Metrics:
    """ Различные метрики качества рекомендательных систем. """

    @staticmethod
    def hit_rate_at_k(
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> float:
        """
        Метрика HitRate@K:
        для какой доли пользователей удалось порекомендовать среди
        первых `k` хотя бы один объект из реального лога.
        Диапазон значений [0, 1], чем выше метрика, тем лучше.

        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида
            `[user_id, item_id, context, relevance]`
        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида
            `[user_id , item_id , timestamp , context , relevance]`
        :param k: какое максимальное количество объектов брать из топа
            рекомендованных для оценки
        """
        top_k_recommendations = (
            recommendations
                .withColumn(
                "relevance_rank",
                sf.row_number().over(
                    Window.partitionBy("user_id", "context")
                        .orderBy(sf.desc("relevance"))
                )
            )
                .filter(sf.col("relevance_rank") <= k)
                .drop("relevance_rank"))

        users_hit = top_k_recommendations.join(
            ground_truth,
            how="inner",
            on=["user_id", "item_id", "context"]
        ).select("user_id").distinct().count()

        users_total = (
            top_k_recommendations.select("user_id").distinct().count()
        )

        return users_hit / users_total

    @staticmethod
    def ndcg_at_k(
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> float:
        """
        Метрика nDCG@k:
        чем релевантнее элементы среди первых `k`, тем выше метрика.
        Диапазон значений [0, 1], чем выше метрика, тем лучше.

        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида
            `[user_id, item_id, context, relevance]`
        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида
            `[user_id , item_id , timestamp , context , relevance]`
        :param k: какое максимальное количество объектов брать из топа
            рекомендованных для оценки
        """
        indexer = (
            StringIndexer(
                inputCol="item_id",
                outputCol="item_idx",
                handleInvalid="keep"
            )
                .fit(ground_truth)
        )

        df_true = indexer.transform(ground_truth)
        df_pred = indexer.transform(recommendations)

        df_pred = (
            df_pred
                .groupby("user_id")
                .agg(sf.collect_list("item_idx").alias("pred_items"))
        )

        df_true = (
            df_true
                .groupby("user_id")
                .agg(sf.collect_list("item_idx").alias("true_items"))
        )

        prediction_and_labels = (
            df_pred
                .join(df_true, ["user_id"], how="inner")
                .rdd
                .map(lambda row: (row[1], row[2]))
        )

        metrics = RankingMetrics(prediction_and_labels)
        return metrics.ndcgAt(k)

    @staticmethod
    def precision_at_k(
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> float:
        """
        precision: точность на `k` первых элементах выдачи

        :param recommendations: выдача рекомендательной системы вида
        (user_id, item_id, context, relevance)
        :param ground_truth: реальный лог действий пользователей
        :param k: какое максимальное количество объектов брать из топа
        рекомендованных для оценки
        """
        indexer = StringIndexer(inputCol="item_id",
                                outputCol="item_idx",
                                handleInvalid='keep') \
            .fit(ground_truth)

        df_true = indexer.transform(ground_truth)
        df_pred = indexer.transform(recommendations)

        df_pred = df_pred \
            .groupby("user_id") \
            .agg(sf.collect_list("item_idx").alias('pred_items'))

        df_true = df_true \
            .groupby("user_id") \
            .agg(sf.collect_list("item_idx").alias('true_items'))

        predictionAndLabels = df_pred \
            .join(df_true, ['user_id'], how='inner') \
            .rdd \
            .map(lambda row: (row[1], row[2]))

        metrics = RankingMetrics(predictionAndLabels)
        return metrics.precisionAt(k)
