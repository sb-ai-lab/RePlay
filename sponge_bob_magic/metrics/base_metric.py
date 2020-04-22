"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Union

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from sponge_bob_magic.constants import CommonDataFrame, IntOrList, NumType
from sponge_bob_magic.converter import convert


class Metric(ABC):
    """ Базовый класс метрик. """

    def __str__(self):
        """ Строковое представление метрики. """
        return type(self).__name__

    def __call__(
        self,
        recommendations: CommonDataFrame,
        ground_truth: CommonDataFrame,
        k: IntOrList,
    ) -> Union[Dict[int, NumType], NumType]:
        """
        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида
            ``[user_id, item_id, relevance]``
        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида
            ``[user_id, item_id, timestamp, relevance]``
        :param k: список индексов, показывающий какое максимальное количество объектов брать из топа
            рекомендованных для оценки
        :return: значение метрики
        """
        recommendations_spark = convert(recommendations)
        ground_truth_spark = convert(ground_truth)
        if not self._check_users(recommendations_spark, ground_truth_spark):
            logger = logging.getLogger("sponge_bob_magic")
            logger.warning(
                "Значение метрики может быть неожиданным:"
                "пользователи в recommendations и ground_truth различаются!"
            )
        return self._get_metric_value(recommendations_spark, ground_truth_spark, k)

    def _get_enriched_recommendations(
        self, recommendations: DataFrame, ground_truth: DataFrame
    ) -> DataFrame:
        """
        Обогащение рекомендаций дополнительной информацией. По умолчанию к рекомендациям добавляется
        столбец, содержащий множество элементов, с которыми взаимодействовал пользователь

        :param recommendations: рекомендации
        :param ground_truth: лог тестовых действий
        :return: рекомендации обогащенные дополнительной информацией
            спарк-датафрейм вида
            ``[user_id, item_id, relevance, *columns]``
        """
        true_items_by_users = ground_truth.groupby("user_id").agg(
            sf.collect_set("item_id").alias("items_id")
        )

        return recommendations.join(true_items_by_users, how="inner", on=["user_id"])

    def _get_metric_value(
        self, recommendations: DataFrame, ground_truth: DataFrame, k: IntOrList
    ) -> Union[Dict[int, NumType], NumType]:
        """
        Расчёт значения метрики

        :param recommendations: рекомендации
        :param ground_truth: лог тестовых действий
        :param k: набор чисел или одно число, по которому рассчитывается метрика
        :return: значения метрики для разных k в виде словаря, если был передан список к,
         иначе просто значение метрики
        """
        if isinstance(k, int):
            k_set = {k}
        else:
            k_set = set(k)
        self.max_k = max(k_set)
        users_count = recommendations.select("user_id").distinct().count()
        agg_fn = self._get_metric_value_by_user

        recs = self._get_enriched_recommendations(recommendations, ground_truth)

        @sf.pandas_udf(
            st.StructType(
                [
                    st.StructField("user_id", recs.schema["user_id"].dataType, True),
                    st.StructField("cum_agg", st.DoubleType(), True),
                    st.StructField("k", st.LongType(), True),
                ]
            ),
            sf.PandasUDFType.GROUPED_MAP,
        )
        def grouped_map(pandas_df):
            pandas_df = (
                pandas_df.sort_values("relevance", ascending=False)
                .reset_index(drop=True)
                .assign(k=pandas_df.index + 1)
            )
            return agg_fn(pandas_df)[["user_id", "cum_agg", "k"]]

        recs = recs.groupby("user_id").apply(grouped_map).where(sf.col("k").isin(k_set))
        total_metric = (
            recs.groupby("k")
            .agg(sf.sum("cum_agg").alias("total_metric"))
            .withColumn("total_metric", sf.col("total_metric") / users_count)
            .select("total_metric", "k")
            .collect()
        )
        res = {row["k"]: row["total_metric"] for row in total_metric}
        if isinstance(k, int):
            res = res[k]
        return res

    @staticmethod
    @abstractmethod
    def _get_metric_value_by_user(pandas_df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчёт значения метрики для каждого пользователя

        :param pandas_df: DataFrame, содержащий рекомендации по каждому пользователю --
            pandas-датафрейм вида ``[user_id, item_id, relevance, k, *columns]``
        :return: DataFrame c рассчитанным полем ``cum_agg`` --
            pandas-датафрейм вида ``[user_id , item_id , cum_agg, *columns]``
        """

    @staticmethod
    def _check_users(recommendations: DataFrame, ground_truth: DataFrame) -> bool:
        """
        Вспомогательный метод, который сравнивает множества пользователей,
        которым выдали рекомендации, и тех, кто есть в тестовых данных

        :param recommendations: рекомендации
        :param ground_truth: лог тестовых действий
        :return: совпадают ли множества пользователей
        """
        left = recommendations.select("user_id").distinct().cache()
        right = ground_truth.select("user_id").distinct().cache()
        left_count = left.count()
        right_count = right.count()
        inner_count = left.join(right, on="user_id").count()
        return left_count == inner_count and right_count == inner_count


class RecOnlyMetric(Metric):
    """Базовый класс для метрик,
    которые измеряют качество списков рекомендаций,
    не сравнивая их с holdout значениями"""

    def __call__(
        self, recommendations: CommonDataFrame, k: IntOrList
    ) -> Union[Dict[int, NumType], NumType]:
        """
        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида
            ``[user_id, item_id, relevance]``
        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида
            ``[user_id, item_id, timestamp, relevance]``
        :param k: список индексов, показывающий какое максимальное количество объектов брать из топа
            рекомендованных для оценки
        :return: значение метрики
        """
        recommendations_spark = convert(recommendations)
        return self._get_metric_value(recommendations_spark, recommendations_spark, k)
