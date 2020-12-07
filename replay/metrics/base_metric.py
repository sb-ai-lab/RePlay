"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from scipy.stats import norm

from replay.constants import AnyDataFrame, IntOrList, NumType
from replay.utils import convert2spark


class Metric(ABC):
    """ Базовый класс метрик. """

    max_k: int

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
            sf.collect_set("item_id").alias("items_id")
        )
        recommendations = recommendations.join(
            true_items_by_users, how="left", on=["user_id"]
        )

        return recommendations.withColumn(
            "items_id",
            sf.coalesce(
                "items_id",
                sf.array().cast(
                    st.ArrayType(ground_truth.schema["item_id"].dataType)
                ),
            ),
        )

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
        if not self._check_users(recommendations_spark, ground_truth_spark):
            logger = logging.getLogger("replay")
            logger.warning(
                "Значение метрики может быть неожиданным: "
                "пользователи в recommendations и ground_truth различаются!"
            )

        if isinstance(k, int):
            k_set = {k}
        else:
            k_set = set(k)
        self.max_k = max(k_set)
        agg_fn = self._get_metric_value_by_user

        recs = self._get_enriched_recommendations(
            recommendations_spark, ground_truth_spark
        )
        max_k = self.max_k

        @sf.pandas_udf(
            st.StructType(
                [
                    st.StructField(
                        "user_id", recs.schema["user_id"].dataType, True
                    ),
                    st.StructField("cum_agg", st.DoubleType(), True),
                    st.StructField("k", st.LongType(), True),
                ]
            ),
            sf.PandasUDFType.GROUPED_MAP,
        )
        def grouped_map(pandas_df):  # pragma: no cover
            additional_rows = max_k - len(pandas_df)
            one_row = pandas_df[
                pandas_df["relevance"] == pandas_df["relevance"].min()
            ].iloc[0]
            one_row["relevance"] = -np.inf
            one_row["item_id"] = np.nan
            if additional_rows > 0:
                pandas_df = pandas_df.append(
                    [one_row] * additional_rows, ignore_index=True
                )
            pandas_df = (
                pandas_df.sort_values("relevance", ascending=False)
                .reset_index(drop=True)
                .assign(k=pandas_df.index + 1)
            )
            return agg_fn(pandas_df)[["user_id", "cum_agg", "k"]]

        distribution = (
            recs.groupby("user_id")
            .apply(grouped_map)
            .where(sf.col("k").isin(k_set))
        )

        return distribution

    @staticmethod
    @abstractmethod
    def _get_metric_value_by_user(pandas_df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчёт значения метрики для каждого пользователя

        :param pandas_df: DataFrame, содержащий рекомендации по каждому
            пользователю -- pandas-датафрейм вида ``[user_id, item_id,
            items_id, k, *columns]``, где
            ``k`` --- порядковый номер рекомендованного объекта ``item_id`` в
            списке рекомендаций для пользоавтеля ``user_id``,
            ``items_id`` --- список объектов, с которыми действительно
            взаимодействовал пользователь в тесте
        :return: DataFrame c рассчитанным полем ``cum_agg`` --
            pandas-датафрейм вида ``[user_id , item_id , cum_agg, *columns]``
        """

    @staticmethod
    def _check_users(
        recommendations: DataFrame, ground_truth: DataFrame
    ) -> bool:
        """
        Вспомогательный метод, который сравнивает множества пользователей,
        которым выдали рекомендации, и тех, кто есть в тестовых данных

        :param recommendations: рекомендации
        :param ground_truth: лог тестовых действий
        :return: совпадают ли множества пользователей
        """
        left = recommendations.select("user_id").distinct().cache()
        right = ground_truth.select("user_id").distinct().cache()
        left_count: int = left.count()
        right_count: int = right.count()
        inner_count: int = left.join(right, on="user_id").count()
        return left_count == inner_count and right_count == inner_count

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
    которые измеряют качество списков рекомендаций,
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
        recommendations_spark = convert2spark(recommendations)
        return self.mean(recommendations_spark, recommendations_spark, k)
