"""
Base classes for quality and diversity metrics.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from scipy.stats import norm

from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, IntOrList, NumType, PandasDataFrame, SparkDataFrame
from replay.utils.session_handler import State
from replay.utils.spark_utils import convert2spark, get_top_k_recs

if PYSPARK_AVAILABLE:
    from pyspark.sql import Column, Window
    from pyspark.sql import functions as sf
    from pyspark.sql import types as st
    from pyspark.sql.column import _to_java_column, _to_seq
    from pyspark.sql.types import DataType


def fill_na_with_empty_array(
    df: SparkDataFrame, col_name: str, element_type: DataType
) -> SparkDataFrame:
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
    ground_truth: DataFrameLike,
    ground_truth_users: Optional[DataFrameLike] = None,
) -> SparkDataFrame:
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


def drop_duplicates(recommendations: DataFrameLike) -> SparkDataFrame:

    """
    Filter duplicated predictions by choosing the most relevant
    """
    return (
        recommendations.withColumn(
            "_num",
            sf.row_number().over(
                Window.partitionBy("user_idx", "item_idx").orderBy(sf.col("relevance").desc())
            ),
        )
        .where(sf.col("_num") == 1)
        .drop("_num")
    )


def filter_sort(recommendations: SparkDataFrame, extra_column: str = None) -> SparkDataFrame:
    """
    Filters duplicated predictions by choosing items with the highest relevance,
    Sorts items in predictions by its relevance,
    If `extra_column` is not None return DataFrame with extra_column e.g. item weight.

    :param recommendations: recommendation list
    :param extra_column: column in recommendations
        which will be return besides ``[user_idx, item_idx]``
    :return: ``[user_idx, item_idx]`` if extra_column = None
        or ``[user_idx, item_idx, extra_column]`` if extra_column exists.
    """
    item_type = recommendations.schema["item_idx"].dataType
    extra_column_type = recommendations.schema[extra_column].dataType if extra_column else None

    recommendations = drop_duplicates(recommendations)

    recommendations = (
        recommendations
        .groupby("user_idx")
        .agg(
            sf.collect_list(
                sf.struct(*[c for c in ["relevance", "item_idx", extra_column] if c is not None]))
            .alias("pred_list"))
        .withColumn("pred_list", sf.reverse(sf.array_sort("pred_list")))
    )

    selection = [
        "user_idx",
        sf.col("pred_list.item_idx")
        .cast(st.ArrayType(item_type, True)).alias("pred")
    ]
    if extra_column:
        selection.append(
            sf.col(f"pred_list.{extra_column}")
            .cast(st.ArrayType(extra_column_type, True)).alias(extra_column)
        )

    recommendations = recommendations.select(*selection)

    return recommendations


def get_enriched_recommendations(
    recommendations: DataFrameLike,
    ground_truth: DataFrameLike,
    max_k: int,
    ground_truth_users: Optional[DataFrameLike] = None,
) -> SparkDataFrame:
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

    true_items_by_users = preprocess_gt(ground_truth, ground_truth_users)
    joined = filter_sort(recommendations).join(
        true_items_by_users, how="right", on=["user_idx"]
    )

    return fill_na_with_empty_array(
        joined, "pred", recommendations.schema["item_idx"].dataType
    )


def process_k(func):
    """Decorator that converts k to list and unpacks result"""

    def wrap(self, recs: SparkDataFrame, k: IntOrList, *args):
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
    _scala_udf_name: Optional[str] = None

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

    @property
    def scala_udf_name(self) -> str:
        """Returns UDF name from `org.apache.spark.replay.utils.ScalaPySparkUDFs`"""
        if self._scala_udf_name:
            return self._scala_udf_name
        else:
            raise NotImplementedError(f"Scala UDF not implemented for {type(self).__name__} class!")

    def __str__(self):
        return type(self).__name__

    def __call__(
        self,
        recommendations: DataFrameLike,
        ground_truth: DataFrameLike,
        k: IntOrList,
        ground_truth_users: Optional[DataFrameLike] = None,
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
    def _conf_interval(self, recs: SparkDataFrame, k_list: list, alpha: float):
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
    def _median(self, recs: SparkDataFrame, k_list: list):
        res = {}
        for k in k_list:
            distribution = self._get_metric_distribution(recs, k)
            value = distribution.agg(
                sf.expr("percentile_approx(value, 0.5)").alias("value")
            ).first()["value"]
            res[k] = value
        return res

    @process_k
    def _mean(self, recs: SparkDataFrame, k_list: list):
        res = {}
        for k in k_list:
            distribution = self._get_metric_distribution(recs, k)
            value = distribution.agg(sf.avg("value").alias("value")).first()[
                "value"
            ]
            res[k] = value
        return res

    def _get_metric_distribution(self, recs: SparkDataFrame, k: int) -> SparkDataFrame:
        """
        :param recs: recommendations
        :param k: depth cut-off
        :return: metric distribution for different cut-offs and users
        """
        if self._use_scala_udf:
            metric_value_col = self.get_scala_udf(
                self.scala_udf_name, [sf.lit(k).alias("k"), *recs.columns[1:]]
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
        log: DataFrameLike,
        recommendations: DataFrameLike,
        ground_truth: DataFrameLike,
        k: IntOrList,
        ground_truth_users: Optional[DataFrameLike] = None,
    ) -> PandasDataFrame:
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
        res = PandasDataFrame()
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
        """
        Returns expression of calling scala UDF as column

        :param udf_name: UDF name from `org.apache.spark.replay.utils.ScalaPySparkUDFs`
        :param params: list of UDF params in right order
        :return: column expression
        """
        sc = State().session.sparkContext  # pylint: disable=invalid-name
        scala_udf = getattr(
            sc._jvm.org.apache.spark.replay.utils.ScalaPySparkUDFs, udf_name
        )()
        return Column(scala_udf.apply(_to_seq(sc, params, _to_java_column)))


# pylint: disable=too-few-public-methods
class RecOnlyMetric(Metric):
    """Base class for metrics that do not need holdout data"""

    @abstractmethod
    def __init__(self, log: DataFrameLike, *args, **kwargs):  # pylint: disable=super-init-not-called
        pass

    # pylint: disable=no-self-use
    @abstractmethod
    def _get_enriched_recommendations(
        self,
        recommendations: DataFrameLike,
        ground_truth: Optional[DataFrameLike],
        max_k: int,
        ground_truth_users: Optional[DataFrameLike] = None,
    ) -> SparkDataFrame:
        pass

    def __call__(
        self,
        recommendations: DataFrameLike,
        k: IntOrList,
        ground_truth_users: Optional[DataFrameLike] = None,
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
    RePlay implements Normalized Capped Importance Sampling for metric calculation with ``NCISMetric`` class.
    This method is mostly applied to RL-based recommendation systems to perform counterfactual evaluation, but could be
    used for any kind of recommender systems. See an article
    `Offline A/B testing for Recommender Systems <http://arxiv.org/abs/1801.07030>` for details.

    *Reward* (metric value for a user-item pair) is weighed by
    the ratio of *current policy score* (current relevance) on *previous policy score* (historical relevance).

    The *weight* is clipped by the *threshold* and put into interval :math:`[\\frac{1}{threshold}, threshold]`.
    Activation function (e.g. softmax, sigmoid) could be applied to the scores before weights calculation.

    Normalization weight for recommended item is calculated as follows:

    .. math::
        w_{ui} = \\frac{f(\pi^t_ui, pi^t_u)}{f(\pi^p_ui, pi^p_u)}

    Where:

    :math:`\pi^t_{ui}` - current policy value (predicted relevance) of the user-item interaction

    :math:`\pi^p_{ui}` - previous policy value (historical relevance) of the user-item interaction.
    Only values for user-item pairs present in current recommendations are used for calculation.

    :math:`\pi_u` - all predicted /historical policy values for selected user :math:`u`

    :math:`f(\pi_{ui}, \pi_u)` - activation function applied to policy values (optional)

    :math:`w_{ui}` - weight of user-item interaction for normalized metric calculation before clipping


    Calculated weights are clipped as follows:

    .. math::
        \hat{w_{ui}} = min(max(\\frac{1}{threshold}, w_{ui}), threshold)

    Normalization metric value for a user is calculated as follows:

    .. math::
        R_u = \\frac{r_{ui} \hat{w_{ui}}}{\sum_{i}\hat{w_{ui}}}

    Where:

    :math:`r_ui` - metric value (reward) for user-item interaction

    :math:`R_u` - metric value (reward) for user :math:`u`

    Weight calculation is implemented in ``_get_enriched_recommendations`` method.
    """

    def __init__(
        self,
        prev_policy_weights: DataFrameLike,
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
    def _softmax_by_user(df: SparkDataFrame, col_name: str) -> SparkDataFrame:
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
    def _sigmoid(df: SparkDataFrame, col_name: str) -> SparkDataFrame:
        """
        Apply sigmoid/logistic function to column `col_name`
        """
        return df.withColumn(
            col_name, sf.lit(1.0) / (sf.lit(1.0) + sf.exp(-sf.col(col_name)))
        )

    @staticmethod
    def _weigh_and_clip(
        df: SparkDataFrame,
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
        recommendations: DataFrameLike,
        ground_truth: DataFrameLike,
        max_k: int,
        ground_truth_users: Optional[DataFrameLike] = None,
    ) -> SparkDataFrame:
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

        recommendations = filter_sort(recommendations, "weight")

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
