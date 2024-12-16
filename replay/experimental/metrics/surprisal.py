from typing import Optional

import numpy as np

from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, SparkDataFrame
from replay.utils.spark_utils import convert2spark, get_top_k_recs

from .base_metric import RecOnlyMetric, fill_na_with_empty_array, filter_sort

if PYSPARK_AVAILABLE:
    from pyspark.sql import (
        functions as sf,
        types as st,
    )


class Surprisal(RecOnlyMetric):
    """
    Measures how many surprising rare items are present in recommendations.

    .. math::
        \\textit{Self-Information}(j)= -\\log_2 \\frac {u_j}{N}

    :math:`u_j` -- number of users that interacted with item :math:`j`.
    Cold items are treated as if they were rated by 1 user.
    That is, if they appear in recommendations it will be completely unexpected.

    Metric is normalized.

    Surprisal for item :math:`j` is

    .. math::
        Surprisal(j)= \\frac {\\textit{Self-Information}(j)}{log_2 N}

    Recommendation list surprisal is the average surprisal of items in it.

    .. math::
        Surprisal@K(i) = \\frac {\\sum_{j=1}^{K}Surprisal(j)} {K}

    Final metric is averaged by users.

    .. math::
        Surprisal@K = \\frac {\\sum_{i=1}^{N}Surprisal@K(i)}{N}
    """

    _scala_udf_name = "getSurprisalMetricValue"

    def __init__(self, log: DataFrameLike, use_scala_udf: bool = False):
        """
        Here we calculate self-information for each item

        :param log: historical data
        """
        self._use_scala_udf = use_scala_udf
        self.log = convert2spark(log)
        n_users = self.log.select("user_idx").distinct().count()
        self.item_weights = self.log.groupby("item_idx").agg(
            (sf.log2(n_users / sf.countDistinct("user_idx")) / np.log2(n_users)).alias("rec_weight")
        )

    @staticmethod
    def _get_metric_value_by_user(k, *args):
        weigths = args[0]
        return sum(weigths[:k]) / k

    def _get_enriched_recommendations(
        self,
        recommendations: SparkDataFrame,
        ground_truth: SparkDataFrame,  # noqa: ARG002
        max_k: int,
        ground_truth_users: Optional[DataFrameLike] = None,
    ) -> SparkDataFrame:
        recommendations = convert2spark(recommendations)
        ground_truth_users = convert2spark(ground_truth_users)
        recommendations = get_top_k_recs(recommendations, max_k)

        recommendations = recommendations.join(self.item_weights, on="item_idx", how="left").fillna(1.0)
        recommendations = filter_sort(recommendations, "rec_weight")
        recommendations = recommendations.select("user_idx", sf.col("rec_weight")).withColumn(
            "rec_weight",
            sf.col("rec_weight").cast(st.ArrayType(st.DoubleType(), True)),
        )
        if ground_truth_users is not None:
            recommendations = fill_na_with_empty_array(
                recommendations.join(ground_truth_users, on="user_idx", how="right"),
                "rec_weight",
                self.item_weights.schema["rec_weight"].dataType,
            )

        return recommendations
