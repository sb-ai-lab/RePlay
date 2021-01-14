"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import operator
from abc import ABC, abstractmethod
from typing import Dict, Union

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from scipy.stats import norm

from replay.constants import AnyDataFrame, IntOrList, NumType
from replay.utils import convert2spark


class Metric(ABC):
    """ Базовый класс метрик. """

    def __str__(self):
        """ Строковое представление метрики. """
        return type(self).__name__

    def __call__(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
    ) -> Union[Dict[int, NumType], NumType]:
        """
        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида ``[user_id, item_id, relevance]``

        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида ``[user_id, item_id, timestamp, relevance]``

        :param k: список индексов, показывающий какое максимальное количество
            объектов брать из топа рекомендованных для оценки

        :return: значение метрики
        """
        return self.mean(recommendations, ground_truth, k)

    def conf_interval(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
        alpha: float = 0.95,
    ) -> Union[Dict[int, NumType], NumType]:
        """Функция возвращает половину ширины доверительного интервала

        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида ``[user_id, item_id, relevance]``

        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида ``[user_id, item_id, timestamp, relevance]``

        :param k: список индексов, показывающий какое максимальное количество
            объектов брать из топа рекомендованных для оценки

        :param alpha: квантиль нормального распределения

        :return: половина ширины доверительного интервала
        """
        distribution = self._get_metric_distribution(
            recommendations, ground_truth, k
        )
        total_metric = (
            distribution.groupby("k")
            .agg(
                sf.stddev("cum_agg").alias("std"),
                sf.count("cum_agg").alias("count"),
            )
            .select(
                sf.when(sf.isnan(sf.col("std")), sf.lit(0.0))
                .otherwise(sf.col("std"))
                .cast("float")
                .alias("std"),
                "count",
                "k",
            )
            .collect()
        )
        quantile = norm.ppf((1 + alpha) / 2)
        res = {
            row["k"]: quantile * row["std"] / (row["count"] ** 0.5)
            for row in total_metric
        }

        return self.unpack_if_int(res, k)

    def median(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
    ) -> Union[Dict[int, NumType], NumType]:
        """Функция возвращает медиану метрики

        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида ``[user_id, item_id, relevance]``

        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида ``[user_id, item_id, timestamp, relevance]``

        :param k: список индексов, показывающий какое максимальное
            количество объектов брать из топа рекомендованных для оценки

        :return: значение медианы
        """
        distribution = self._get_metric_distribution(
            recommendations, ground_truth, k
        )
        total_metric = (
            distribution.groupby("k")
            .agg(
                sf.expr("percentile_approx(cum_agg, 0.5)").alias(
                    "total_metric"
                )
            )
            .select("total_metric", "k")
            .collect()
        )
        res = {row["k"]: row["total_metric"] for row in total_metric}
        return self.unpack_if_int(res, k)

    # pylint: disable=missing-docstring
    @staticmethod
    def unpack_if_int(res: Dict, k: IntOrList) -> Union[Dict, float]:
        if isinstance(k, int):
            return res[k]
        return res

    def mean(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
    ) -> Union[Dict[int, NumType], NumType]:
        """Функция возвращает среднее значение метрики

        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида  ``[user_id, item_id, relevance]``

        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида ``[user_id, item_id, timestamp, relevance]``

        :param k: список индексов, показывающий какое максимальное
            количество объектов брать из топа рекомендованных для оценки

        :return: среднее значение
        """
        distribution = self._get_metric_distribution(
            recommendations, ground_truth, k
        )
        total_metric = (
            distribution.groupby("k")
            .agg(sf.avg("cum_agg").alias("total_metric"))
            .select("total_metric", "k")
            .collect()
        )
        res = {row["k"]: row["total_metric"] for row in total_metric}
        return self.unpack_if_int(res, k)

    # pylint: disable=no-self-use
    def _get_enriched_recommendations(
        self, recommendations: DataFrame, ground_truth: DataFrame
    ) -> DataFrame:
        """
        Обогащение рекомендаций дополнительной информацией.
        По умолчанию к рекомендациям добавляется столбец,
        содержащий множество элементов,
        с которыми взаимодействовал пользователь

        :param recommendations: рекомендации
        :param ground_truth: лог тестовых действий
        :return: рекомендации обогащенные дополнительной информацией
            спарк-датафрейм вида ``[user_id, item_id, relevance, *columns]``
        """
        true_items_by_users = ground_truth.groupby("user_id").agg(
            sf.collect_set("item_id").alias("ground_truth")
        )
        sort_udf = sf.udf(
            self._sorter,
            returnType=st.ArrayType(ground_truth.schema["item_id"].dataType),
        )
        recommendations = (
            recommendations.groupby("user_id")
            .agg(
                sf.collect_list(sf.struct("relevance", "item_id")).alias(
                    "pred"
                )
            )
            .select("user_id", sort_udf(sf.col("pred")).alias("pred"))
            .join(true_items_by_users, how="left", on=["user_id"])
        )

        return recommendations.withColumn(
            "ground_truth",
            sf.coalesce(
                "ground_truth",
                sf.array().cast(
                    st.ArrayType(ground_truth.schema["item_id"].dataType)
                ),
            ),
        )

    @staticmethod
    def _sorter(items):
        res = sorted(items, key=operator.itemgetter(0), reverse=True)
        return [item[1] for item in res]

    def _get_metric_distribution(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
    ) -> DataFrame:
        """
        Распределение метрики

        :param recommendations: рекомендации
        :param ground_truth: лог тестовых действий
        :param k: набор чисел или одно число, по которому рассчитывается метрика
        :return: распределение значения метрики для разных k по пользователям
        """
        recommendations_spark = convert2spark(recommendations)
        ground_truth_spark = convert2spark(ground_truth)

        if isinstance(k, int):
            k_set = {k}
        else:
            k_set = set(k)
        recs = self._get_enriched_recommendations(
            recommendations_spark, ground_truth_spark
        )
        cur_class = self.__class__
        distribution = recs.rdd.flatMap(
            # pylint: disable=protected-access
            lambda x: cur_class._get_metric_value_by_user_all_k(k_set, *x)
        ).toDF(
            "user_id {}, cum_agg double, k long".format(
                recs.schema["user_id"].dataType.typeName()
            )
        )

        return distribution

    @classmethod
    def _get_metric_value_by_user_all_k(cls, k_set, user_id, *args):
        """
        Расчёт значения метрики для каждого пользователя для нескольких k

        :param k_set: набор чисел, для которых расчитывается метрика,
        :param user_id: идентификатор пользователя,
        :param *args: дополнительные параметры, необходимые для расчета
            метрики. Перечень параметров совпадает со списком столбцов
            датафрейма, который возвращает метод '''_get_enriched_recommendations'''
        :return: значение метрики для данного пользователя
        """
        result = []
        for k in k_set:
            result.append(
                (
                    user_id,
                    # pylint: disable=no-value-for-parameter
                    cls._get_metric_value_by_user(k, *args),
                    k,
                )
            )
        return result

    @staticmethod
    @abstractmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        """
        Расчёт значения метрики для каждого пользователя

        :param k: число, для которого расчитывается метрика,
        :param pred: список объектов, рекомендованных пользователю
        :param ground_truth: список объектов, с которыми действительно
        взаимодействовал пользователь в тесте
        :return: значение метрики для данного пользователя
        """

    def user_distribution(
        self,
        log: AnyDataFrame,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
    ) -> pd.DataFrame:
        """
        Получить среднее значение метрики для пользователей с данным количеством оценок.

        :param log: датафрейм с логом оценок для подсчета количества оценок у пользователей
        :param recommendations: датафрейм с рекомендациями
        :param ground_truth: тестовые данные
        :param k: сколько брать айтемов из рекомендаций
        :return: пандас датафрейм
        """
        log = convert2spark(log)
        count = log.groupBy("user_id").count()
        dist = self._get_metric_distribution(recommendations, ground_truth, k)
        res = count.join(dist, on="user_id")
        res = (
            res.groupBy("k", "count")
            .agg(sf.avg("cum_agg").alias("value"))
            .orderBy(["k", "count"])
            .select("k", "count", "value")
            .toPandas()
        )
        return res


# pylint: disable=too-few-public-methods
class RecOnlyMetric(Metric):
    """Базовый класс для метрик,
    которые измеряют качество рекомендаций,
    не сравнивая их с holdout значениями"""

    @abstractmethod
    def __init__(self, log: AnyDataFrame, *args, **kwargs):
        pass

    def __call__(  # type: ignore
        self, recommendations: AnyDataFrame, k: IntOrList
    ) -> Union[Dict[int, NumType], NumType]:
        """
        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида ``[user_id, item_id, relevance]``

        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида ``[user_id, item_id, timestamp, relevance]``

        :param k: список индексов, показывающий какое максимальное количество
        объектов брать из топа рекомендованных для оценки

        :return: значение метрики
        """
        return self.mean(recommendations, recommendations, k)

    @staticmethod
    @abstractmethod
    def _get_metric_value_by_user(k, *args) -> float:
        """
        Расчёт значения метрики для каждого пользователя

        :param k: число, для которого расчитывается метрика,
        :param pred: список объектов, рекомендованных пользователю
        :param *args: дополнительные параметры, необходимые для расчета
            метрики. Перечень параметров совпадает со списком столбцов
            датафрейма, который возвращает метод '''_get_enriched_recommendations'''
        :return: значение метрики для данного пользователя
        """
