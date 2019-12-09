"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.rdd import RDD
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf


class Metrics:
    """ Различные метрики качества рекомендательных систем. """

    @staticmethod
    def _merge_prediction_and_truth(recommendations: DataFrame,
                                    ground_truth: DataFrame) -> RDD:
        """
        Вспомогательный метод-трансформер,
        который джойнит входную истинную выдачу с выдачей модели.
        Нужен для использование встроенных метрик RankingMetrics.

        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида
            `[user_id, item_id, context, relevance]`
        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида
            `[user_id , item_id , timestamp , context , relevance]`
        :return: rdd-датафрейм, который является джойном входных датафреймов,
            пригодный для использования в RankingMetrics (после map)
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
        window = Window.partitionBy('user_id').orderBy(
            sf.col('relevance').desc()
        )
        df_pred = (df_pred
                   .withColumn('pred_items',
                               sf.collect_list('item_idx').over(window))
                   .groupby("user_id")
                   .agg(sf.max('pred_items').alias('pred_items')))
        df_true = (df_true
                   .withColumn('true_items',
                               sf.collect_list('item_idx').over(window))
                   .groupby("user_id")
                   .agg(sf.max('true_items').alias('true_items')))

        prediction_and_labels = (
            df_pred
            .join(df_true, ["user_id"], how="inner")
            .rdd
        )

        return prediction_and_labels

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
            recommendations.withColumn(
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
        dataframe = Metrics._merge_prediction_and_truth(
            recommendations, ground_truth
        )
        dataframe = dataframe.map(lambda row: (row[1], row[2]))
        metrics = RankingMetrics(dataframe)
        return metrics.ndcgAt(k)

    @staticmethod
    def precision_at_k(
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> float:
        """
        Метрика Precision@k:
        точность на `k` первых элементах выдачи.
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
        dataframe = Metrics._merge_prediction_and_truth(
            recommendations, ground_truth
        )
        dataframe = dataframe.map(lambda row: (row[1], row[2]))
        metrics = RankingMetrics(dataframe)
        return metrics.precisionAt(k)

    @staticmethod
    def map_at_k(
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> float:
        """
        Метрика MAP@k (mean average precision):
        средняя точность на `k` первых элементах выдачи.
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
        dataframe = Metrics._merge_prediction_and_truth(
            recommendations, ground_truth
        )
        dataframe = dataframe.map(lambda row: (row[1][:k], row[2][:k]))
        metrics = RankingMetrics(dataframe)
        return metrics.meanAveragePrecision
