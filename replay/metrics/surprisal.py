from functools import partial

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from replay.constants import AnyDataFrame
from replay.utils import convert2spark
from replay.metrics.base_metric import RecOnlyMetric, sorter


# pylint: disable=too-few-public-methods
class Surprisal(RecOnlyMetric):
    """
    Measures how many surprising rare items are present in recommendations.

    .. math::
        \\textit{Self-Information}(j)= -\log_2 \\frac {u_j}{N}

    :math:`u_j` -- number of users that interacted with item :math:`j`.
    Cold items are treated as if they were rated by 1 user.
    That is, if they appear in recommendations it will be completely unexpected.

    Metric is normalized.

    Surprisal for item :math:`j` is

    .. math::
        Surprisal(j)= \\frac {\\textit{Self-Information}(j)}{log_2 N}

    Recommendation list surprisal is the average surprisal of items in it.

    .. math::
        Surprisal@K(i) = \\frac {\sum_{j=1}^{K}Surprisal(j)} {K}

    Final metric is averaged by users.

    .. math::
        Surprisal@K = \\frac {\sum_{i=1}^{N}Surprisal@K(i)}{N}
    """

    def __init__(
        self, log: AnyDataFrame
    ):  # pylint: disable=super-init-not-called
        """
        Here we calculate self-information for each item

        :param log: historical data
        """
        self.log = convert2spark(log)
        n_users = self.log.select("user_id").distinct().count()  # type: ignore
        self.item_weights = self.log.groupby("item_id").agg(
            (
                sf.log2(n_users / sf.countDistinct("user_id"))  # type: ignore
                / np.log2(n_users)
            ).alias("rec_weight")
        )

    @staticmethod
    def _get_metric_value_by_user(k, *args):
        weigths = args[0]
        return sum(weigths[:k]) / k

    def _get_enriched_recommendations(
        self, recommendations: DataFrame, ground_truth: DataFrame
    ) -> DataFrame:
        recommendations = convert2spark(recommendations)
        sort_udf = sf.udf(
            partial(sorter, index=2),
            returnType=st.ArrayType(st.DoubleType()),
        )
        return (
            recommendations.join(self.item_weights, on="item_id", how="left")
            .fillna(1)
            .groupby("user_id")
            .agg(
                sf.collect_list(
                    sf.struct("relevance", "item_id", "rec_weight")
                ).alias("rec_weight")
            )
            .select(
                "user_id", sort_udf(sf.col("rec_weight")).alias("rec_weight")
            )
        )
