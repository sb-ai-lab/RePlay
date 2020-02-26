"""
Метрики необходимы для определния качества модели.
Для рассчета большинства метрик требуются таблица с рекомендациями и таблица с реальными значениями - списком айтемов, с которыми провзаимодействовал пользователь.
Все метрики рассчитываются не для общего списка рекомендаций, а только для первых top@k, k задается в качестве параметра.

"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from sponge_bob_magic.constants import IterOrList
from sponge_bob_magic.converter import convert

NumType = Union[int, float]


class Metric(ABC):
    """ Базовый класс метрик. """
    def __init__(self, log: DataFrame = None):
        pass

    def __call__(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
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
        if not self._check_users(recommendations, ground_truth):
            logging.warning(
                "Значение метрики может быть неожиданным:"
                "пользователи в recommendations и ground_truth различаются!"
            )
        return self._get_metric_value(recommendations, ground_truth, k)

    def _get_enriched_recommendations(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame
    ) -> DataFrame:
        """
        Расчет весов для items и добавление их к рекомендациям

        :param recommendations: рекомендации
        :param ground_truth: лог тестовых действий
        :return: рекомендации с весами items
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

        :param pandas_df: DataFrame, содержащий рекомендации по каждому пользователю
        :return: DataFrame c рассчитанным полем cum_agg
        """

    @abstractmethod
    def __str__(self):
        """ Строковое представление метрики. """

    def _check_users(
            self,
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
    первых ``k`` хотя бы один объект из реального лога.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
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
    чем релевантнее элементы среди первых ``k``, тем выше метрика.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
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
    точность на ``k`` первых элементах выдачи.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
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
    средняя точность на ``k`` первых элементах выдачи.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
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
    какую долю объектов из реального лога мы покажем в рекомендациях среди
    первых ``k`` (в среднем по пользователям).
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
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
        return "Surprisal"

    def __init__(self, log: DataFrame):
        log = convert(log)
        n_users = log.select("user_id").distinct().count()
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
