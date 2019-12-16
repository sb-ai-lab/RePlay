"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from abc import abstractmethod, ABC
from typing import Union

from pyspark.ml.feature import StringIndexer
from pyspark.rdd import RDD
from pyspark.sql import DataFrame, Window, SparkSession
from pyspark.sql import functions as sf

NumType = Union[int, float]


class Metrics(ABC):
    """ Базовый класс метрик. """

    def __init__(self, spark: SparkSession, **kwargs):
        """
        :param spark: инициализированная спарк-сессия
        """
        self.spark = spark

    def __call__(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        """
        Call-метод, позволяет вызывать подсчет метрики сразу от инстанса
        класса. Тип метрики определяется классом-наследником.

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
        return self.calculate(recommendations, ground_truth, k)

    @abstractmethod
    def calculate(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        """
        Абстрактный метод, который должны реализовать классы-наследники.

        Считает метрику. Тип метрики определяется классом-наследником.

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
