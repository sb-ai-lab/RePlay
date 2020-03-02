"""
Для рассчета большинства метрик требуется таблица с рекомендациями и таблица с реальными значениями -- списком айтемов,
с которыми провзаимодействовал пользователь.

Все метрики рассчитываются для первых ``K`` объектов в рекомендации.
Поддерживается возможность рассчета метрик сразу по нескольким ``K``,
в таком случае будет возвращен словарь с результатами, а не число.

Если реализованных метрик недостаточно, библиотека поддерживает возможность
:ref:`добавления своих метрик <new-metric>`.
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
from sponge_bob_magic.models import PopRec
from sponge_bob_magic.models.base_rec import Recommender

NumType = Union[int, float]
CommonDataFrame = Union[DataFrame, pd.DataFrame]


class Metric(ABC):
    """ Базовый класс метрик. """

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
        self.max_k = max(k_set)
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
            return agg_fn(pandas_df)[["user_id","cum_agg","k"]]

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
    Доля пользователей, для которой хотя бы одна рекомендация из
    первых ``K`` была успешна.

    .. math::
        HitRate@K(i) = \max_{j \in [1..K]}\mathbb{1}_{r_{ij}}

    .. math::
        HitRate@K = \\frac {\sum_{i=1}^{N}HitRate@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- индикатор взаимодействия пользователя :math:`i` с рекомендацией :math:`j`
"""

    def __init__(self):
        pass

    def __str__(self):
        return "HitRate"

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        pandas_df = pandas_df.assign(is_good_item=pandas_df[["item_id", "items_id"]]
                                     .apply(lambda x: int(x["item_id"] in x["items_id"]), 1))

        return pandas_df.assign(cum_agg=pandas_df.is_good_item.cummax())


class NDCG(Metric):
    """
    Normalized Discounted Cumulative Gain учитывает порядок в списке рекомендаций --
    чем ближе к началу списка полезные рекомендации, тем больше значение метрики.

    Реализован бинарный вариант релевантности -- был объект или нет,
    не произвольная шкала полезности вроде оценок.

    Метрика определяется следующим образом:

    .. math::
        DCG@K(i) = \sum_{j=1}^{K}\\frac{\mathbb{1}_{r_{ij}}}{\log_2 (j+1)}


    :math:`\\mathbb{1}_{r_{ij}}` -- индикатор взаимодействия пользователя :math:`i` с рекомендацией :math:`j`

    Для перехода от :math:`DCG` к :math:`nDCG` необходимо подсчитать максимальное значение метрики для пользователя :math:`i` и  длины рекомендаций :math:`K`

    .. math::
        IDCG@K(i) = max(DCG@K(i)) = \sum_{j=1}^{K}\\frac{\mathbb{1}_{j\le|Rel_i|}}{\log_2 (j+1)}

    .. math::
        nDCG@K(i) = \\frac {DCG@K(i)}{IDCG@K(i)}

    :math:`|Rel_i|` -- количество элементов, с которыми пользовтель :math:`i` взаимодействовал

    Для расчета итоговой метрики усредняем по всем пользователям

    .. math::
        nDCG@K = \\frac {\sum_{i=1}^{N}nDCG@K(i)}{N}
    """

    def __init__(self):
        pass

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
    Средняя доля успешных рекомендаций среди первых ``K`` элементов выдачи.

    .. math::
        Precision@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij}}}{K}

    .. math::
        Precision@K = \\frac {\sum_{i=1}^{N}Precision@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- индикатор взаимодействия пользователя :math:`i` с рекомендацией :math:`j`
"""

    def __init__(self):
        pass

    def __str__(self):
        return "Precision"

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        pandas_df = pandas_df.assign(is_good_item=pandas_df[["item_id", "items_id"]]
                                     .apply(lambda x: int(x["item_id"] in x["items_id"]), 1))

        return pandas_df.assign(cum_agg=pandas_df["is_good_item"].cumsum() / pandas_df.k)


class MAP(Metric):
    """
    Mean Average Precision -- усреднение ``Precision`` по целым числам от 1 до ``K``, усреднённое по пользователям.

    .. math::
        &AP@K(i) = \\frac 1K \sum_{j=1}^{K}\mathbb{1}_{r_{ij}}Precision@j(i)

        &MAP@K = \\frac {\sum_{i=1}^{N}AP@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- индикатор взаимодействия пользователя :math:`i` с рекомендацией :math:`j`
    """

    def __init__(self):
        pass

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
    Какая доля объектов, с которыми взаимодействовал пользователь в тестовых данных, была показана ему в списке рекомендаций длины ``K``?

    .. math::
        Recall@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij}}}{|Rel_i|}

    .. math::
        Recall@K = \\frac {\sum_{i=1}^{N}Recall@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- индикатор взаимодействия пользователя :math:`i` с рекомендацией :math:`j`

    :math:`|Rel_i|` -- количество элементов, с которыми взаимодействовал пользователь :math:`i`
    """

    def __init__(self):
        pass

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
    Показывает насколько редкие предметы выдаются в рекомендациях.
    В качестве оценки редкости используется собственная информация объекта,
    превращающая популярность объекта в его неожиданность.

    .. math::
        \\textit{Self-Information}(j)= -\log_2 \\frac {u_j}{N}

    :math:`u_j` -- количество пользователей, которые взаимодействовали с объектом :math:`j`.
    Для холодных объектов количество взаимодействий считается равным 1,
    то есть их появление в рекомендациях будет считаться крайне неожиданным.

    Чтобы метрику было проще интерпретировать, это значение нормируется.

    Таким образом редкость объекта :math:`j` определяется как

    .. math::
        Surprisal(j)= \\frac {\\textit{Self-Information}(j)}{log_2 N}

    Для списка рекомендаций длины :math:`K` значение метрики определяется как среднее значение редкости.

    .. math::
        Surprisal@K(i) = \\frac {\sum_{j=1}^{K}Surprisal(j)} {K}

    Итоговое значение усредняется по пользователям

    .. math::
        Surprisal@K = \\frac {\sum_{i=1}^{N}Surprisal@K(i)}{N}
    """

    def __str__(self):
        return "Surprisal"

    def __init__(self, log: CommonDataFrame):
        """
        Чтобы посчитать метрику, необходимо предрассчитать собственную информацию каждого объекта.

        :param log: датафрейм с логом действий пользователей
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

class Unexpectedness(Metric):
    """
    Доля объектов в рекомендациях, которая не содержится в рекомендациях некоторого базового алгоритма.
    По умолчанию используется рекомендатель по популярности ``PopRec``.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"user_id": [1, 1, 2, 3], "item_id": [1, 2, 1, 3], "relevance": [5, 5, 5, 5], "timestamp": [1, 1, 1, 1], "context": [1, 1, 1, 1]})
    >>> dd = pd.DataFrame({"user_id": [1, 2, 1, 2], "item_id": [1, 2, 3, 1], "relevance": [5, 5, 5, 5], "timestamp": [1, 1, 1, 1], "context": [1, 1, 1, 1]})
    >>> m = Unexpectedness(df)
    >>> m(dd, dd, [1, 2])
    {1: 1.0, 2: 0.25}


    Возможен так же режим, в котором рекомендации базового алгоритма передаются сразу при инициализации и рекомендатель не обучается

    >>> de = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [1, 2, 3], "relevance": [5, 5, 5], "timestamp": [1, 1, 1], "context": [1, 1, 1]})
    >>> dr = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [0, 0, 1], "relevance": [5, 5, 5], "timestamp": [1, 1, 1], "context": [1, 1, 1]})
    >>> m = Unexpectedness(dr, None)
    >>> round(m(de, de, 3), 2)
    0.67
    """

    def __init__(self, log: CommonDataFrame, rec: Recommender = PopRec()):
        """
        Есть два варианта инициализации в зависимости от значения параметра ``rec``.
        Если ``rec`` -- рекомендатель, то ``log`` считается данными для обучения.
        Если ``rec is None``, то ``log`` считается готовыми предсказаниями какой-то внешней модели,
        с которой необходимо сравниться.

        :param log: пандас или спарк датафрейм
        :param rec: одна из проинициализированных моделей библиотеки, либо ``None``
        """

        # super().__init__(log=log)
        self.log = convert(log)
        if rec is None:
            self.train_model = False
        else:
            self.train_model = True
            rec.fit(log=self.log)
            self.model = rec

    def __str__(self):
        return "Unexpectedness"

    def __call__(
            self,
            recommendations: CommonDataFrame,
            ground_truth: CommonDataFrame,
            k: IterOrList
    ) -> Union[Dict[int, NumType], NumType]:
        return super().__call__(recommendations, recommendations, k)

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        recs = pandas_df["item_id"]
        pandas_df["cum_agg"] = pandas_df.apply(
            lambda row:
            (
                 row["k"] -
                 np.isin(
                     recs[:row["k"]],
                     row["items_id"][:row["k"]]
                 ).sum()
             )/row["k"],
            axis=1)
        return pandas_df

    def _get_enriched_recommendations(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame
    ) -> DataFrame:
        if self.train_model:
            pred = self.model.predict(log=self.log, k=self.max_k)
        else:
            pred = self.log
        items_by_users = (pred
            .groupby("user_id").agg(
            sf.collect_list("item_id").alias("items_id")))
        res = recommendations.join(
            items_by_users,
            how="inner",
            on=["user_id"]
        )
        return res
