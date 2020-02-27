"""
Метрики необходимы для определния качества модели.
Для рассчета большинства метрик требуются таблица с рекомендациями и таблица с реальными значениями - списком айтемов,
с которыми провзаимодействовал пользователь.
Все метрики рассчитываются не для общего списка рекомендаций, а только для первых top@K,
``K`` задается в качестве параметра. Возможно передавать как набор разных K, так и одно значение

Реализованы следующие метрики

- :ref:`HitRate <HitRate>`
- :ref:`Precision <Precision>`
- :ref:`MAP <MAP>`
- :ref:`Recall <Recall>`
- :ref:`NDCG <NDCG>`
- :ref:`Surprisal <Surprisal>`

В случае, если указанных метрик недостаточно, библиотека поддерживает возможность
:ref:`создания новых метрик <new-metric>`.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from sponge_bob_magic.constants import IterOrList
from sponge_bob_magic.converter import convert

NumType = Union[int, float]
CommonDataFrame = Union[DataFrame, pd.DataFrame]


class Metric(ABC):
    """ Базовый класс метрик. """

    def __init__(self,
                 log: Optional[CommonDataFrame] = None,
                 user_features: Optional[CommonDataFrame] = None,
                 item_features: Optional[CommonDataFrame] = None
                 ):
        if log:
            self.log = convert(log)

        if user_features:
            self.user_features = convert(user_features)

        if item_features:
            self.item_features = convert(item_features)

    def __call__(
            self,
            recommendations: CommonDataFrame,
            ground_truth: CommonDataFrame,
            k: IterOrList
    ) -> Union[Dict[int, NumType], NumType]:
        """
        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида
            ``[user_id, item_id, context, relevance]``
        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида
            ``[user_id , item_id , timestamp , context , relevance]``
        :param k: список индексов, показывающий какое максимальное количество объектов брать из топа
            рекомендованных для оценки
        :return: значение метрики
        """
        recommendations_spark = convert(recommendations)
        ground_truth_spark = convert(ground_truth)
        if not self._check_users(recommendations_spark, ground_truth_spark):
            logging.warning(
                    "Значение метрики может быть неожиданным:"
                    "пользователи в recommendations и ground_truth различаются!"
            )
        return self._get_metric_value(recommendations_spark, ground_truth_spark, k)

    def _get_enriched_recommendations(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame
    ) -> DataFrame:
        """
        Обогащение рекомендаций дополнительной информацией. По умолчанию к рекомендациям добавляется
        столбец, содержащий множество элементов, с которыми взаимодействовал пользователь

        :param recommendations: рекомендации
        :param ground_truth: лог тестовых действий
        :return: рекомендации обогащенные дополнительной информацией
            спарк-датафрейм вида
            ``[user_id , item_id , context , relevance, *columns]``
        """
        true_items_by_users = (ground_truth
                               .groupby("user_id").agg(
                                   sf.collect_set("item_id").alias("items_id")))

        return recommendations.join(
            true_items_by_users,
            how="inner",
            on=["user_id"]
        )

    def _get_metric_value(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: IterOrList
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
        users_count = recommendations.select("user_id").distinct().count()
        agg_fn = self._get_metric_value_by_user

        @sf.pandas_udf(st.StructType([st.StructField("user_id", st.StringType(), True),
                                      st.StructField("cum_agg", st.DoubleType(), True),
                                      st.StructField("k", st.LongType(), True)
                                      ]),
                       sf.PandasUDFType.GROUPED_MAP)
        def grouped_map(pandas_df):
            pandas_df = (pandas_df.sort_values("relevance", ascending=False)
                         .reset_index(drop=True)
                         .assign(k=pandas_df.index + 1))
            return agg_fn(pandas_df)[["user_id", "cum_agg", "k"]]

        recs = self._get_enriched_recommendations(recommendations, ground_truth)
        recs = recs.groupby("user_id").apply(grouped_map).where(sf.col("k").isin(k_set))
        total_metric = (recs
                        .groupby("k").agg(sf.sum("cum_agg").alias("total_metric"))
                        .withColumn("total_metric", sf.col("total_metric") / users_count)
                        .select("total_metric", "k").collect())

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
            pandas-датафрейм вида ``[user_id , item_id , context , relevance, k, *columns]``
        :return: DataFrame c рассчитанным полем ``cum_agg`` --
            pandas-датафрейм вида ``[user_id , item_id , cum_agg, *columns]``
        """

    @abstractmethod
    def __str__(self):
        """ Строковое представление метрики. """

    @staticmethod
    def _check_users(
            recommendations: DataFrame,
            ground_truth: DataFrame
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
        left_count = left.count()
        right_count = right.count()
        inner_count = left.join(right, on="user_id").count()
        return left_count == inner_count and right_count == inner_count


class HitRate(Metric):
    """
    Метрика HitRate@K:
    для какой доли пользователей удалось порекомендовать среди
    первых ``K`` хотя бы один объект из реального лога.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.

    Для каждого пользователя метрика считается следующим образом

    .. math::
        HitRate@K(i) = \max_{j \in [1..K]}\mathbb{1}_{r_{ij}}

    Здесь :math:`r_{ij}` - индикатор было ли взаимодействие с рекомендацией :math:`j` у пользователя :math:`i`.

    Для расчета метрики по всем пользователям достаточно усреднить её значение

    .. math::
        HitRate@K = \\frac {\sum_{i=1}^{N}HitRate@K(i)}{N}
"""

    def __str__(self):
        return "HitRate"

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        pandas_df = pandas_df.assign(is_good_item=pandas_df[["item_id", "items_id"]]
                                     .apply(lambda x: int(x["item_id"] in x["items_id"]), 1))

        return pandas_df.assign(cum_agg=pandas_df.is_good_item.cummax())


class NDCG(Metric):
    """
    Метрика nDCG@k:
    чем релевантнее элементы среди первых ``K``, тем выше метрика.
    Здесь реализован бинарный вариант релевантности а не произвольный из диапазона [0, 1].
    Диапазон значений [0, 1], чем выше метрика, тем лучше.

    Для каждого пользователя метрика считается следующим образом

    .. math::
        &DCG@K(i) = \sum_{j=1}^{K}\\frac{\mathbb{1}_{r_{ij}}}{\log_2 (j+1)}

        &IDCG@K(i) = \sum_{j=1}^{K}\\frac{\mathbb{1}_{j\le|Rel_i|}}{\log_2 (j+1)}

        &nDCG@K(i) = \\frac {DCG@K(i)}{IDCG@K(i)}

    Здесь :math:`r_{ij}` - индикатор было ли взаимодействие с рекомендацией :math:`j` у пользователя :math:`i`,
    :math:`|Rel_i|` - количество элементов, с которыми пользовтель :math:`i` взаимодействовал

    Для расчета метрики по всем пользователям достаточно усреднить её значение

    .. math::
        nDCG@K = \\frac {\sum_{i=1}^{N}nDCG@K(i)}{N}
    """

    def __str__(self):
        return "nDCG"

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        pandas_df = pandas_df.assign(is_good_item=pandas_df[["item_id", "items_id"]]
                                     .apply(lambda x: int(x["item_id"] in x["items_id"]), 1))
        pandas_df = pandas_df.assign(
            sorted_good_item=pandas_df["k"].le(pandas_df["items_id"].str.len()))

        return pandas_df.assign(
            cum_agg=(pandas_df["is_good_item"] / np.log2(pandas_df.k + 1)).cumsum() /
            (pandas_df["sorted_good_item"] / np.log2(pandas_df.k + 1)).cumsum())


class Precision(Metric):
    """
    Метрика Precision@k:
    точность на ``K`` первых элементах выдачи.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.

    Для каждого пользователя метрика считается следующим образом

    .. math::
        Precision@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij}}}{K}

    Здесь :math:`r_{ij}` - индикатор было ли взаимодействие с рекомендацией :math:`j` у пользователя :math:`i`.

    Для расчета метрики по всем пользователям достаточно усреднить её значение

    .. math::
        Precision@K = \\frac {\sum_{i=1}^{N}Precision@K(i)}{N}
"""

    def __str__(self):
        return "Precision"

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        pandas_df = pandas_df.assign(is_good_item=pandas_df[["item_id", "items_id"]]
                                     .apply(lambda x: int(x["item_id"] in x["items_id"]), 1))

        return pandas_df.assign(cum_agg=pandas_df["is_good_item"].cumsum() / pandas_df.k)


class MAP(Metric):
    """
    Метрика MAP@k (mean average precision):
    средняя точность на ``K`` первых элементах выдачи.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.

    Для каждого пользователя метрика считается следующим образом

    .. math::
        &AP@K(i) = \\frac 1K \sum_{j=1}^{K}\mathbb{1}_{r_{ij}}Precision@j_i

        &MAP@K(i) = \\frac 1K \sum_{j=1}^{K}AP@k_i

    Здесь :math:`r_{ij}` - индикатор было ли взаимодействие с рекомендацией :math:`j` у пользователя :math:`i`

    Для расчета метрики по всем пользователям достаточно усреднить её значение

    .. math::
        MAP@K = \\frac {\sum_{i=1}^{N}MAP@K(i)}{N}
    """

    def __str__(self):
        return "MAP"

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        pandas_df = pandas_df.assign(
            is_good_item=pandas_df[["item_id", "items_id"]].apply(
                lambda x: int(x["item_id"] in x["items_id"]), 1),
            good_items_count=pandas_df["items_id"].str.len())

        return pandas_df.assign(cum_agg=(pandas_df["is_good_item"].cumsum()
                                         * pandas_df["is_good_item"]
                                         / pandas_df.k
                                         / pandas_df[["k", "good_items_count"]].min(axis=1))
                                .cumsum())


class Recall(Metric):
    """
    Метрика Recall@K:
    доля объектов из реального лога, которые мы покажем в рекомендациях среди
    первых ``K`` (в среднем по пользователям).
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    Для каждого пользователя метрика считается следующим образом

    .. math::
        Recall@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij}}}{|Rel_i|}

    Здесь :math:`r_{ij}` - индикатор было ли взаимодействие с рекомендацией :math:`j` у пользователя :math:`i`,
    :math:`|Rel_i|` - количество элементов, с которыми пользовтель :math:`i` взаимодействовал

    Для расчета метрики по всем пользователям достаточно усреднить её значение

    .. math::
        Recall@K = \\frac {\sum_{i=1}^{N}Recall@K(i)}{N}
    """

    def __str__(self):
        return "Recall"

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        pandas_df = pandas_df.assign(is_good_item=pandas_df[["item_id", "items_id"]]
                                     .apply(lambda x: int(x["item_id"] in x["items_id"]), 1))

        return pandas_df.assign(
            cum_agg=pandas_df["is_good_item"].cumsum() / pandas_df["items_id"].str.len())


class Surprisal(Metric):
    """
    Метрика Surprisal@k:
    суммарная популярность объектов, показаных пользовтелю в рекомендациях среди
    первых ``K`` (в среднем по пользователям).
    Чем выше метрика, тем больше непопулярных объектов попадают в рекомендации.
    Для каждого пользователя метрика считается следующим образом

    .. math::
        &Weight_{ij}= -\log_2 \\frac {u_j}{N}

        &Weight_{ij,norm}= \\frac {Weight_ij}{log_2 N}

        &Surprisal@K(i) = \sum_{j=1}^{K}Weight_{ij,norm}

    Здесь :math:`u_j` - количество пользователей, которе взаимодействовали с элементом :math:`j`.
    Для холодных объектов количество взаимодействий считается равным 1.

    Для расчета метрики по всем пользователям достаточно усреднить её значение

    .. math::
        Surprisal@K = \\frac {\sum_{i=1}^{N}Surprisal@K(i)}{N}
    """

    def __str__(self):
        return "Surprisal"

    def __init__(self, log: CommonDataFrame):
        """
        Предрасчет весов всех items, в зависимости от их популярности
        """
        super().__init__(log=log)
        n_users = self.log.select("user_id").distinct().count()
        self.item_weights = log.groupby("item_id").agg(
                (sf.log2(n_users / sf.countDistinct("user_id"))
                 / np.log2(n_users)).alias("rec_weight"))

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        return pandas_df.assign(cum_agg=pandas_df["rec_weight"].cumsum() / pandas_df["k"])

    def _get_enriched_recommendations(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame
    ) -> DataFrame:
        return (recommendations.join(self.item_weights,
                                     on="item_id",
                                     how="left").fillna(1))
