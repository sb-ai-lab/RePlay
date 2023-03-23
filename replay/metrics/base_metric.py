"""
Base classes for quality and diversity metrics.
"""
import logging
import operator
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional

import pandas as pd
from pyspark.sql import Column, SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from pyspark.sql.types import DataType
from pyspark.sql import Window
from pyspark.sql.column import _to_java_column, _to_seq
from scipy.stats import norm

from replay.constants import AnyDataFrame, IntOrList, NumType
from replay.utils import convert2spark, get_top_k_recs


# pylint: disable=no-member
def sorter(
    items: Tuple[Tuple], extra_position=None
) -> Union[List, Tuple[List, List]]:
    """
    Sorts a list of tuples and chooses unique objects.
    Sorting is made using relevance values (descending).
    Unique items from `item_idx` column are selected.
    If `extra_position` is None, only list with ordered items is returned.
    Otherwise two lists are returned, the first is with item_ids
    and the second is with the tuples' elements on 'extra_position', e.g item weight.

    :param items: tuples ``(relevance, item_idx, *args)``.
    :param extra_position: index of the element in tuple to be returned in addition to item_idx,
        if None, only item_ids are returned
    :return: list of unique item_ids sorted by relevance is descending order
        and additional list of corresponding values on `extra_position` if not None
    """
    res = sorted(items, key=operator.itemgetter(0), reverse=True)
    set_res = set()
    item_ids = []
    extra_values = []
    for item in res:
        if item[1] not in set_res:
            set_res.add(item[1])
            item_ids.append(item[1])
            if extra_position is not None:
                extra_values.append(item[extra_position])
    if extra_position is None:
        return item_ids
    return item_ids, extra_values


def fill_na_with_empty_array(
    df: DataFrame, col_name: str, element_type: DataType
) -> DataFrame:
    """
    Fill empty values in array column with empty array of `element_type` values.
    :param df: dataframe with `col_name` column of ArrayType(`element_type`)
    :param col_name: name of a column to fill missing values
    :param element_type: DataType of an array element
    :return: df with `col_name` na values filled with empty arrays
    """
    return df.withColumn(
        col_name,
        sf.coalesce(
            col_name,
            sf.array().cast(st.ArrayType(element_type)),
        ),
    )


def preprocess_gt(
    ground_truth: AnyDataFrame,
    ground_truth_users: Optional[AnyDataFrame] = None,
) -> DataFrame:
    """
    Preprocess `ground_truth` data before metric calculation
    :param ground_truth: spark dataframe with columns ``[user_idx, item_idx, relevance]``
    :param ground_truth_users: spark dataframe with column ``[user_idx]``
    :return: spark dataframe with columns ``[user_idx, ground_truth]``
    """
    ground_truth = convert2spark(ground_truth)
    ground_truth_users = convert2spark(ground_truth_users)

    true_items_by_users = ground_truth.groupby("user_idx").agg(
        sf.collect_set("item_idx").alias("ground_truth")
    )
    if ground_truth_users is not None:
        true_items_by_users = true_items_by_users.join(
            ground_truth_users, on="user_idx", how="right"
        )
        true_items_by_users = fill_na_with_empty_array(
            true_items_by_users,
            "ground_truth",
            ground_truth.schema["item_idx"].dataType,
        )

    return true_items_by_users


def get_enriched_recommendations(
    recommendations: AnyDataFrame,
    ground_truth: AnyDataFrame,
    max_k: int,
    ground_truth_users: Optional[AnyDataFrame] = None,
) -> DataFrame:
    """
    Leave max_k recommendations for each user,
    merge recommendations and ground truth into a single DataFrame
    and aggregate items into lists so that each user has only one record.

    :param recommendations: recommendation list
    :param ground_truth: test data
    :param max_k: maximal k value to calculate the metric for.
        `max_k` most relevant predictions are left for each user
    :param ground_truth_users: list of users to consider in metric calculation.
        if None, only the users from ground_truth are considered.
    :return:  ``[user_idx, pred, ground_truth]``
    """
    recommendations = convert2spark(recommendations)
    # if there are duplicates in recommendations,
    # we will leave fewer than k recommendations after sort_udf
    recommendations = get_top_k_recs(recommendations, k=max_k)
    sort_udf = sf.udf(
        sorter,
        returnType=st.ArrayType(recommendations.schema["item_idx"].dataType),
    )

    true_items_by_users = preprocess_gt(ground_truth, ground_truth_users)
    joined = (
        recommendations.groupby("user_idx")
        .agg(sf.collect_list(sf.struct("relevance", "item_idx")).alias("pred"))
        .select("user_idx", sort_udf(sf.col("pred")).alias("pred"))
        .join(true_items_by_users, how="right", on=["user_idx"])
    )

    return fill_na_with_empty_array(
        joined, "pred", recommendations.schema["item_idx"].dataType
    )


def process_k(func):
    """Decorator that converts k to list and unpacks result"""

    def wrap(self, recs: DataFrame, k: IntOrList, *args):
        if isinstance(k, int):
            k_list = [k]
        else:
            k_list = k

        res = func(self, recs, k_list, *args)

        if isinstance(k, int):
            return res[k]
        return res

    return wrap


class Metric(ABC):
    """Base metric class"""

    _logger: Optional[logging.Logger] = None

    def __init__(self, use_scala_udf: bool = False) -> None:
        self._use_scala_udf = use_scala_udf

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    def __str__(self):
        return type(self).__name__

    def __call__(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
        ground_truth_users: Optional[AnyDataFrame] = None,
    ) -> Union[Dict[int, NumType], NumType]:
        """
        :param recommendations: model predictions in a
            DataFrame ``[user_idx, item_idx, relevance]``
        :param ground_truth: test data
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: depth cut-off. Truncates recommendation lists to top-k items.
        :param ground_truth_users: list of users to consider in metric calculation.
            if None, only the users from ground_truth are considered.
        :return: metric value
        """
        recs = get_enriched_recommendations(
            recommendations,
            ground_truth,
            max_k=k if isinstance(k, int) else max(k),
            ground_truth_users=ground_truth_users,
        )
        return self._mean(recs, k)

    @process_k
    def _conf_interval(self, recs: DataFrame, k_list: list, alpha: float):
        res = {}
        quantile = norm.ppf((1 + alpha) / 2)
        for k in k_list:
            distribution = self._get_metric_distribution(recs, k)
            value = (
                distribution.agg(
                    sf.stddev("value").alias("std"),
                    sf.count("value").alias("count"),
                )
                .select(
                    sf.when(
                        sf.isnan(sf.col("std")) | sf.col("std").isNull(),
                        sf.lit(0.0),
                    )
                    .otherwise(sf.col("std"))
                    .cast("float")
                    .alias("std"),
                    "count",
                )
                .first()
            )
            res[k] = quantile * value["std"] / (value["count"] ** 0.5)
        return res

    @process_k
    def _median(self, recs: DataFrame, k_list: list):
        res = {}
        for k in k_list:
            distribution = self._get_metric_distribution(recs, k)
            value = distribution.agg(
                sf.expr("percentile_approx(value, 0.5)").alias("value")
            ).first()["value"]
            res[k] = value
        return res

    @process_k
    def _mean(self, recs: DataFrame, k_list: list):
        res = {}
        for k in k_list:
            distribution = self._get_metric_distribution(recs, k)
            value = distribution.agg(sf.avg("value").alias("value")).first()[
                "value"
            ]
            res[k] = value
        return res

    def _get_metric_distribution(self, recs: DataFrame, k: int) -> DataFrame:
        """
        :param recs: recommendations
        :param k: depth cut-off
        :return: metric distribution for different cut-offs and users
        """
        if self._use_scala_udf:
            # Possibly bad approach to define column names for udf call
            # because we don't know columns ordering
            # and we don't know exactly what is columns in recs
            cols = [col for col in recs.columns if col != "user_idx"]
            metric_value_col = self._get_metric_value_by_user_scala_udf(
                sf.lit(k).alias("k"), *cols
            ).alias("value")
            return recs.select("user_idx", metric_value_col)

        cur_class = self.__class__
        distribution = recs.rdd.flatMap(
            # pylint: disable=protected-access
            lambda x: [
                (x[0], float(cur_class._get_metric_value_by_user(k, *x[1:])))
            ]
        ).toDF(
            f"user_idx {recs.schema['user_idx'].dataType.typeName()}, value double"
        )
        return distribution

    @staticmethod
    @abstractmethod
    def _get_metric_value_by_user_scala_udf(
        k: Union[str, Column],
        pred: Union[str, Column],
        ground_truth: Union[str, Column],
    ) -> Column:
        """Returns scala udf that calcs metric for one user as Column"""

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

    # pylint: disable=too-many-arguments
    def user_distribution(
        self,
        log: AnyDataFrame,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
        ground_truth_users: Optional[AnyDataFrame] = None,
    ) -> pd.DataFrame:
        """
        Get mean value of metric for all users with the same number of ratings.

        :param log: history DataFrame to calculate number of ratings per user
        :param recommendations: prediction DataFrame
        :param ground_truth: test data
        :param k: depth cut-off
        :param ground_truth_users: list of users to consider in metric calculation.
            if None, only the users from ground_truth are considered.
        :return: pandas DataFrame
        """
        log = convert2spark(log)
        count = log.groupBy("user_idx").count()
        if hasattr(self, "_get_enriched_recommendations"):
            recs = self._get_enriched_recommendations(
                recommendations,
                ground_truth,
                max_k=k if isinstance(k, int) else max(k),
                ground_truth_users=ground_truth_users,
            )
        else:
            recs = get_enriched_recommendations(
                recommendations,
                ground_truth,
                max_k=k if isinstance(k, int) else max(k),
                ground_truth_users=ground_truth_users,
            )
        if isinstance(k, int):
            k_list = [k]
        else:
            k_list = k
        res = pd.DataFrame()
        for cut_off in k_list:
            dist = self._get_metric_distribution(recs, cut_off)
            val = count.join(dist, on="user_idx", how="right").fillna(
                0, subset="count"
            )
            val = (
                val.groupBy("count")
                .agg(sf.avg("value").alias("value"))
                .orderBy(["count"])
                .select("count", "value")
                .toPandas()
            )
            res = res.append(val, ignore_index=True)
        return res

    @staticmethod
    def get_scala_udf(udf_name: str, params: List) -> Column:
        """Returns expression of calling scala UDF as column

        Args:
            udf_name: UDF name from 'org.apache.spark.replay.utils.ScalaPySparkUDFs'
            params: list of UDF params in right order

        Returns: column expression

        """
        sc = (  # pylint: disable=invalid-name
            SparkSession.getActiveSession().sparkContext
        )
        scala_udf = getattr(
            sc._jvm.org.apache.spark.replay.utils.ScalaPySparkUDFs, udf_name
        )()
        return Column(scala_udf.apply(_to_seq(sc, params, _to_java_column)))


# pylint: disable=too-few-public-methods
class RecOnlyMetric(Metric):
    """Base class for metrics that do not need holdout data"""

    @abstractmethod
    def __init__(self, log: AnyDataFrame, *args, **kwargs):
        pass

    # pylint: disable=no-self-use
    @abstractmethod
    def _get_enriched_recommendations(
        self,
        recommendations: AnyDataFrame,
        ground_truth: Optional[AnyDataFrame],
        max_k: int,
        ground_truth_users: Optional[AnyDataFrame] = None,
    ) -> DataFrame:
        pass

    def __call__(
        self,
        recommendations: AnyDataFrame,
        k: IntOrList,
        ground_truth_users: Optional[AnyDataFrame] = None,
    ) -> Union[Dict[int, NumType], NumType]:
        """
        :param recommendations: predictions of a model,
            DataFrame  ``[user_idx, item_idx, relevance]``
        :param k: depth cut-off
        :param ground_truth_users: list of users to consider in metric calculation.
            if None, only the users from ground_truth are considered.
        :return: metric value
        """
        recs = self._get_enriched_recommendations(
            recommendations,
            None,
            max_k=k if isinstance(k, int) else max(k),
            ground_truth_users=ground_truth_users,
        )
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


class NCISMetric(Metric):
    """
    Normalized capped importance sampling, where each recommendation is being weighted
    by the ratio of current policy score on previous policy score.
    The weight is also capped by some threshold value.

    Source: arxiv.org/abs/1801.07030
    """

    def __init__(
        self,
        prev_policy_weights: AnyDataFrame,
        threshold: float = 10.0,
        activation: Optional[str] = None,
        use_scala_udf: bool = False,
    ):  # pylint: disable=super-init-not-called
        """
        :param prev_policy_weights: historical item of user-item relevance (previous policy values)
        :threshold: capping threshold, applied after activation,
            relevance values are cropped to interval [1/`threshold`, `threshold`]
        :activation: activation function, applied over relevance values.
            "logit"/"sigmoid", "softmax" or None
        """
        self._use_scala_udf = use_scala_udf
        self.prev_policy_weights = convert2spark(
            prev_policy_weights
        ).withColumnRenamed("relevance", "prev_relevance")
        self.threshold = threshold
        if activation is None or activation in ("logit", "sigmoid", "softmax"):
            self.activation = activation
            if activation == "softmax":
                self.logger.info(
                    "For accurate softmax calculation pass only one `k` value "
                    "in the NCISMetric metrics `call`"
                )
        else:
            raise ValueError(f"Unexpected `activation` - {activation}")
        if threshold <= 0:
            raise ValueError("Threshold should be positive real number")

    @staticmethod
    def _softmax_by_user(df: DataFrame, col_name: str) -> DataFrame:
        """
        Subtract minimal value (relevance) by user from `col_name`
        and apply softmax by user to `col_name`.
        """
        return (
            df.withColumn(
                "_min_rel_user",
                sf.min(col_name).over(Window.partitionBy("user_idx")),
            )
            .withColumn(
                col_name, sf.exp(sf.col(col_name) - sf.col("_min_rel_user"))
            )
            .withColumn(
                col_name,
                sf.col(col_name)
                / sf.sum(col_name).over(Window.partitionBy("user_idx")),
            )
            .drop("_min_rel_user")
        )

    @staticmethod
    def _sigmoid(df: DataFrame, col_name: str) -> DataFrame:
        """
        Apply sigmoid/logistic function to column `col_name`
        """
        return df.withColumn(
            col_name, sf.lit(1.0) / (sf.lit(1.0) + sf.exp(-sf.col(col_name)))
        )

    @staticmethod
    def _weigh_and_clip(
        df: DataFrame,
        threshold: float,
        target_policy_col: str = "relevance",
        prev_policy_col: str = "prev_relevance",
    ):
        """
        Clip weights to fit into interval [1/threshold, threshold].
        """
        lower, upper = 1 / threshold, threshold
        return (
            df.withColumn(
                "weight_unbounded",
                sf.col(target_policy_col) / sf.col(prev_policy_col),
            )
            .withColumn(
                "weight",
                sf.when(sf.col(prev_policy_col) == sf.lit(0.0), sf.lit(upper))
                .when(
                    sf.col("weight_unbounded") < sf.lit(lower), sf.lit(lower)
                )
                .when(
                    sf.col("weight_unbounded") > sf.lit(upper), sf.lit(upper)
                )
                .otherwise(sf.col("weight_unbounded")),
            )
            .select("user_idx", "item_idx", "relevance", "weight")
        )

    def _reweighing(self, recommendations):
        if self.activation == "softmax":
            recommendations = self._softmax_by_user(
                recommendations, col_name="prev_relevance"
            )
            recommendations = self._softmax_by_user(
                recommendations, col_name="relevance"
            )
        elif self.activation in ["logit", "sigmoid"]:
            recommendations = self._sigmoid(
                recommendations, col_name="prev_relevance"
            )
            recommendations = self._sigmoid(
                recommendations, col_name="relevance"
            )

        return self._weigh_and_clip(recommendations, self.threshold)

    def _get_enriched_recommendations(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        max_k: int,
        ground_truth_users: Optional[AnyDataFrame] = None,
    ) -> DataFrame:
        """
        Merge recommendations and ground truth into a single DataFrame
        and aggregate items into lists so that each user has only one record.

        :param recommendations: recommendation list
        :param ground_truth: test data
        :param max_k: maximal k value to calculate the metric for.
            `max_k` most relevant predictions are left for each user
        :param ground_truth_users: list of users to consider in metric calculation.
            if None, only the users from ground_truth are considered.
        :return:  ``[user_idx, pred, ground_truth]``
        """
        recommendations = convert2spark(recommendations)
        ground_truth = convert2spark(ground_truth)
        ground_truth_users = convert2spark(ground_truth_users)

        true_items_by_users = ground_truth.groupby("user_idx").agg(
            sf.collect_set("item_idx").alias("ground_truth")
        )

        group_on = ["item_idx"]
        if "user_idx" in self.prev_policy_weights.columns:
            group_on.append("user_idx")
        recommendations = get_top_k_recs(recommendations, k=max_k)

        recommendations = recommendations.join(
            self.prev_policy_weights, on=group_on, how="left"
        ).na.fill(0.0, subset=["prev_relevance"])

        recommendations = self._reweighing(recommendations)

        weight_type = recommendations.schema["weight"].dataType
        item_type = ground_truth.schema["item_idx"].dataType

        sort_ids_weights_udf = sf.udf(
            lambda x: sorter(items=x, extra_position=2),
            returnType=st.StructType(
                [
                    st.StructField("pred", st.ArrayType(item_type)),
                    st.StructField("weight", st.ArrayType(weight_type)),
                ]
            ),
        )

        recommendations = (
            recommendations.groupby("user_idx")
            .agg(
                sf.collect_list(
                    sf.struct("relevance", "item_idx", "weight")
                ).alias("rel_id_weight")
            )
            .withColumn(
                "pred_weight",
                sort_ids_weights_udf(sf.col("rel_id_weight")),
            )
            .select(
                "user_idx",
                sf.col("pred_weight.pred"),
                sf.col("pred_weight.weight"),
            )
        )
        if ground_truth_users is not None:
            true_items_by_users = true_items_by_users.join(
                ground_truth_users, on="user_idx", how="right"
            )

        recommendations = recommendations.join(
            true_items_by_users, how="right", on=["user_idx"]
        )
        return fill_na_with_empty_array(
            fill_na_with_empty_array(recommendations, "pred", item_type),
            "weight",
            weight_type,
        )
