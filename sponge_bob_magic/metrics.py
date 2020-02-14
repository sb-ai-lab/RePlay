"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from abc import ABC, abstractmethod
from math import log2
from typing import Union, Iterable, Dict, Callable

from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.rdd import RDD
from pyspark.sql import DataFrame, Window, Column
from pyspark.sql import functions as sf

from sponge_bob_magic.utils import get_top_k_recs

NumType = Union[int, float]


class Metric(ABC):
    """ Базовый класс метрик. """

    def __call__(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: Iterable[int]
    ) -> NumType:
        """
        :param recommendations: выдача рекомендательной системы,
            спарк-датарейм вида
            `[user_id, item_id, context, relevance]`
        :param ground_truth: реальный лог действий пользователей,
            спарк-датафрейм вида
            `[user_id , item_id , timestamp , context , relevance]`
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

    @abstractmethod
    def _get_metric_value(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: Iterable[int]
    ) -> NumType:
        """
        Расчёт значения метрики
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
            спарк-датафрейм вида
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

    @staticmethod
    def _get_metrics_diff_top_k(metric_elements: DataFrame,
                                k: Iterable[int],
                                agg_fn: Callable[[Column], Column]
                                ) -> Dict[int, NumType]:
        """
            Рассчитывает усредненную метрику по различным топ-k рекомендаций из набора.

            :param metric_elements: элеменеты ряда для расчета метрики, спарк-датафрейм с колонками
                `[user_id , item_id , context , metric]`
            :param k: набор количеств рекомендаций, по которым необходимо подсчитать метрики
            :return: топ-k рекомендации, словарь с key - количество рекомендаций и value - значение метрики
        """
        k_set = set(k)
        window = (Window
                  .partitionBy(metric_elements["user_id"])
                  .orderBy(metric_elements["relevance"].desc()))

        cum_agg = (metric_elements
                   .withColumn('k',
                               sf.count(metric_elements.relevance).over(window))
                   .withColumn('cum_agg',
                               agg_fn(metric_elements.metric).over(window))
                   .where(sf.col("k").isin(k_set)))
        total_metric = (cum_agg
                        .groupby("k").agg(
            sf.avg("cum_agg").alias("total_metric"))
                        .select("total_metric", "k")).collect()

        return {row["k"]: row["total_metric"] for row in total_metric}

    @staticmethod
    def _merge_prediction_and_truth_top_k(
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> DataFrame:
        top_k_recommendations = get_top_k_recs(recommendations, k)
        true_items_by_users = (ground_truth
            .groupby("user_id").agg(
                sf.collect_set("item_id").alias("items_id")))

        @sf.udf("int")
        def check_is_in(item_id,items_id):
            return int(item_id in items_id)

        merged_dataframe = top_k_recommendations.join(
            true_items_by_users,
            how="inner",
            on=["user_id"]
        ).withColumn("metric", check_is_in(sf.col("item_id"),sf.col("items_id")))

        return merged_dataframe


class HitRate(Metric):
    """
    Метрика HitRate@K:
    для какой доли пользователей удалось порекомендовать среди
    первых `k` хотя бы один объект из реального лога.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def __str__(self):
        return "HitRate"

    def _get_metric_value(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: Iterable[int]
    ) -> Dict[int, NumType]:
        dataframe = self._merge_prediction_and_truth_top_k(recommendations, ground_truth, max(k))

        return self._get_metrics_diff_top_k(dataframe, k, sf.max)


class NDCG(Metric):
    """
    Метрика nDCG@k:
    чем релевантнее элементы среди первых `k`, тем выше метрика.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def __str__(self):
        return "nDCG"

    def _get_metric_value(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: Iterable[int]
    ) -> Dict[int, NumType]:
        dataframe = NDCG._merge_prediction_and_truth_top_k(recommendations, ground_truth, max(k))

        #@sf.pandas_udf("double", sf.PandasUDFType.GROUPED_MAP)
        @sf.udf("double")
        def mean_udf(v):
            return v.mean()

        return self._get_metrics_diff_top_k(dataframe, k, mean_udf)

        # dataframe = self._merge_prediction_and_truth(
        #     recommendations, ground_truth
        # )
        #
        # def _calc_metric_elem(row):
        #     true_items_set = set(row[2])
        #     return [{"user_id": row["user_id"], "item_id": item_id, "metric": int(item_id in true_items_set),
        #              "relevance": ind} for ind, item_id in enumerate(row["pred_items"])]
        #
        # dataframe = dataframe.flatMap(_calc_metric_elem).toDF()
        # # dataframe = dataframe.map(lambda row: (row[1], row[2]))
        # # print(dataframe.take(10))
        # # metrics = RankingMetrics(dataframe)
        # return dataframe  # metrics.ndcgAt(k)


class Precision(Metric):
    """
    Метрика Precision@k:
    точность на `k` первых элементах выдачи.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def __str__(self):
        return "Precision"

    def _get_metric_value(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        dataframe = self._merge_prediction_and_truth(
            recommendations, ground_truth
        )
        dataframe = dataframe.map(lambda row: (row[1], row[2]))
        metrics = RankingMetrics(dataframe)
        return metrics.precisionAt(k)


class MAP(Metric):
    """
    Метрика MAP@k (mean average precision):
    средняя точность на `k` первых элементах выдачи.
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def __str__(self):
        return "MAP"

    def _get_metric_value(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        dataframe = self._merge_prediction_and_truth(
            recommendations, ground_truth
        )
        dataframe = dataframe.map(lambda row: (row[1][:k], row[2][:k]))
        metrics = RankingMetrics(dataframe)
        return metrics.meanAveragePrecision


class Recall(Metric):
    """
    Метрика Recall@K:
    какую долю объектов из реального лога мы покажем в рекомендациях среди
    первых `k` (в среднем по пользователям).
    Диапазон значений [0, 1], чем выше метрика, тем лучше.
    """

    def __str__(self):
        return "Recall"

    def _get_metric_value(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        top_k_recommendations = get_top_k_recs(recommendations, k)
        hits = top_k_recommendations.join(
            ground_truth,
            how="inner",
            on=["user_id", "item_id"]
        ).groupby("user_id").agg(sf.count("item_id").alias("hits"))
        totals = (
            ground_truth
                .groupby("user_id")
                .agg(sf.count("item_id").alias("totals"))
        )
        total_recall, total_users = (
            totals.join(hits, on=["user_id"], how="left")
                .withColumn(
                "recall",
                sf.coalesce(sf.col("hits") / sf.col("totals"), sf.lit(0))
            )
                .agg(
                sf.sum("recall").alias("total_hits"),
                sf.count("recall").alias("total_users")
            )
                .select("total_hits", "total_users")
                .head()[:2]
        )
        return total_recall / total_users


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

    def __init__(self,
                 log: DataFrame,
                 normalize: bool = False):
        """
        Считает популярность и собственную информацию каждого объета.

        :param log: спарк-датафрейм вида
            `[user_id, item_id, timestamp, context, relevance]`;
            содержит информацию о взаимодействии пользователей с объектами
        """
        n_users = log.select("user_id").distinct().count()
        max_value = -log2(1 / n_users)
        stats = log.groupby("item_id").agg(
            sf.countDistinct("user_id").alias("count")
        )

        stats = stats.withColumn("popularity", stats["count"] / n_users)
        stats = stats.withColumn("self-information", -sf.log2("popularity"))
        stats = stats.withColumn(
            "normalized_si",
            stats["self-information"] / max_value
        )

        self.stats = stats
        self.normalize = normalize
        self.fill_value = 1.0 if normalize else max_value

    def _get_metric_value(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            k: int
    ) -> NumType:
        metric = "normalized_si" if self.normalize else "self-information"

        self_information = self.stats.select(["item_id", metric])
        top_k_recommendations = get_top_k_recs(recommendations, k)

        recs = top_k_recommendations.join(self_information,
                                          on="item_id",
                                          how="left").fillna(self.fill_value)
        list_mean = (
            recs
                .groupby("user_id")
                .agg(sf.mean(metric).alias(metric))
        )

        global_mean = (
            list_mean
                .select(sf.mean(metric).alias("mean"))
                .collect()[0]["mean"]
        )

        return global_mean
