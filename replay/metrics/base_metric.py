"""
Base classes for quality and diversity metrics.
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


# pylint: disable=no-member
def sorter(items, index=1):
    """Sorts a list of tuples and chooses unique objects.

    :param items: tuples ``(relevance, item_id, *args)``.
        Sorting is made using relevance values and unique items
        are selected using element at ``index``'s position.
    :param index: index of the element in tuple to be returned
    :return: unique sorted elements
    """
    res = sorted(items, key=operator.itemgetter(0), reverse=True)
    set_res = set()
    list_res = []
    for item in res:
        if item[1] not in set_res:
            set_res.add(item[1])
            list_res.append(item[index])
    return list_res


def get_enriched_recommendations(
    recommendations: AnyDataFrame, ground_truth: AnyDataFrame
) -> DataFrame:
    """
    Adds additional info to recommendations.
    By default adds column containing number of elements user interacted with.

    :param recommendations: recommendation list
    :param ground_truth: test data
    :return: recommendations with additional columns,
        spark DataFrame ``[user_id, item_id, relevance, *columns]``
    """
    recommendations = convert2spark(recommendations)
    ground_truth = convert2spark(ground_truth)
    true_items_by_users = ground_truth.groupby("user_id").agg(
        sf.collect_set("item_id").alias("ground_truth")
    )
    sort_udf = sf.udf(
        sorter,
        returnType=st.ArrayType(ground_truth.schema["item_id"].dataType),
    )
    recommendations = (
        recommendations.groupby("user_id")
        .agg(sf.collect_list(sf.struct("relevance", "item_id")).alias("pred"))
        .select("user_id", sort_udf(sf.col("pred")).alias("pred"))
        .join(true_items_by_users, how="right", on=["user_id"])
    )

    return recommendations.withColumn(
        "pred",
        sf.coalesce(
            "pred",
            sf.array().cast(
                st.ArrayType(ground_truth.schema["item_id"].dataType)
            ),
        ),
    )


class Metric(ABC):
    """Base metric class"""

    def __str__(self):
        return type(self).__name__

    def __call__(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
    ) -> Union[Dict[int, NumType], NumType]:
        """
        :param recommendations: model predictions in a
            DataFrame ``[user_id, item_id, relevance]``
        :param ground_truth: test data
            ``[user_id, item_id, timestamp, relevance]``
        :param k: depth cut-off. Truncates recommendation lists to top-k items.
        :return: metric value
        """
        recs = get_enriched_recommendations(recommendations, ground_truth)
        return self._mean(recs, k)

    def _conf_interval(self, recs: DataFrame, k: IntOrList, alpha: float):
        distribution = self._get_metric_distribution(recs, k)
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

        return self._unpack_if_int(res, k)

    def _median(self, recs: DataFrame, k: IntOrList):
        distribution = self._get_metric_distribution(recs, k)
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
        return self._unpack_if_int(res, k)

    @staticmethod
    def _unpack_if_int(res: Dict, k: IntOrList) -> Union[Dict, float]:
        if isinstance(k, int):
            return res[k]
        return res

    def _mean(self, recs: DataFrame, k: IntOrList):
        distribution = self._get_metric_distribution(recs, k)
        total_metric = (
            distribution.groupby("k")
            .agg(sf.avg("cum_agg").alias("total_metric"))
            .select("total_metric", "k")
            .collect()
        )
        res = {row["k"]: row["total_metric"] for row in total_metric}
        return self._unpack_if_int(res, k)

    def _get_metric_distribution(
        self,
        recs: DataFrame,
        k: IntOrList,
    ) -> DataFrame:
        """
        :param recs: recommendations
        :param k: one or more depth cut-offs
        :return: metric distribution for different cut-offs and users
        """

        if isinstance(k, int):
            k_set = {k}
        else:
            k_set = set(k)
        cur_class = self.__class__
        distribution = recs.rdd.flatMap(
            # pylint: disable=protected-access
            lambda x: cur_class._get_metric_value_by_user_all_k(k_set, *x)
        ).toDF(
            f"""user_id {recs.schema["user_id"].dataType.typeName()},
cum_agg double, k long"""
        )

        return distribution

    @classmethod
    def _get_metric_value_by_user_all_k(cls, k_set, user_id, *args):
        """
        Calculate metric using multiple depth cut-offs.

        :param k_set: depth cut-offs
        :param user_id: user identificator
        :param *args: extra parameters, returned by
            '''self._get_enriched_recommendations''' method
        :return: metric values for all users at each cut-off
        """
        result = []
        for k in k_set:
            result.append(
                (
                    user_id,
                    # pylint: disable=no-value-for-parameter
                    float(cls._get_metric_value_by_user(k, *args)),
                    k,
                )
            )
        return result

    @staticmethod
    @abstractmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        """
        Metric calculation for one user.

        :param k: depth cut-off
        :param pred: recommendations
        :param ground_truth: test data
        :return: metric value for current user
        """

    def user_distribution(
        self,
        log: AnyDataFrame,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
    ) -> pd.DataFrame:
        """
        Get mean value of metric for all users with the same number of ratings.

        :param log: history DataFrame to calculate number of ratings per user
        :param recommendations: prediction DataFrame
        :param ground_truth: test data
        :param k: depth cut-off
        :return: pandas DataFrame
        """
        log = convert2spark(log)
        count = log.groupBy("user_id").count()
        if hasattr(self, "_get_enriched_recommendations"):
            recs = self._get_enriched_recommendations(
                recommendations, ground_truth
            )
        else:
            recs = get_enriched_recommendations(recommendations, ground_truth)
        dist = self._get_metric_distribution(recs, k)
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
    """Base class for metrics that do not need holdout data"""

    @abstractmethod
    def __init__(self, log: AnyDataFrame, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def _get_enriched_recommendations(
        recommendations: AnyDataFrame, ground_truth: AnyDataFrame
    ) -> DataFrame:
        pass

    def __call__(  # type: ignore
        self, recommendations: AnyDataFrame, k: IntOrList
    ) -> Union[Dict[int, NumType], NumType]:
        """
        :param recommendations: predictions of a model,
            DataFrame  ``[user_id, item_id, relevance]``
        :param k: depth cut-off
        :return: metric value
        """
        recs = self._get_enriched_recommendations(recommendations, None)
        return self._mean(recs, k)

    @staticmethod
    @abstractmethod
    def _get_metric_value_by_user(k, *args) -> float:
        """
        Metric calculation for one user.

        :param k: depth cut-off
        :param *args: extra parameters, returned by
            '''self._get_enriched_recommendations''' method
        :return: metric value for current user
        """
