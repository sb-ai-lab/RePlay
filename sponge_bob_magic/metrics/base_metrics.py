"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from abc import ABC, abstractmethod
from typing import Union

from pyspark.ml.feature import StringIndexer
from pyspark.rdd import RDD
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf

NumType = Union[int, float]


class Metric(ABC):
    """ Базовый класс метрик. """

    @abstractmethod
    def __call__(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        """
        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида
            `[user_id, item_id, context, relevance]`
        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида
            `[user_id , item_id , timestamp , context , relevance]`
        :param k: какое максимальное количество объектов брать из топа
            рекомендованных для оценки
        :return: значение метрики
        """

    @abstractmethod
    def __str__(self):
        """ Строковое представление метрики. """

    @staticmethod
    def _merge_prediction_and_truth(
            recommendations: DataFrame,
            ground_truth: DataFrame
    ) -> RDD:
        """
        Вспомогательный метод-трансформер,
        который джойнит входную истинную выдачу с выдачей модели.
        Нужен для использование встроенных метрик RankingMetrics
        (см. `pyspark.mllib.evaluation.RankingMetrics`).

        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида
            `[user_id, item_id, context, relevance]`
        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида
            `[user_id , item_id , timestamp , context , relevance]`
        :return: rdd-датафрейм, который является джойном входных датафреймов,
            пригодный для использования в RankingMetrics
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
