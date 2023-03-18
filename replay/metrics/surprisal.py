from functools import partial
from typing import Optional, Union

import numpy as np
from pyspark.sql import Column
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from replay.constants import AnyDataFrame
from replay.metrics.base_metric import (
    fill_na_with_empty_array,
    RecOnlyMetric,
    sorter,
)
from replay.utils import convert2spark, get_top_k_recs


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
        self, log: AnyDataFrame,
        use_scala_udf: bool = False
    ):  # pylint: disable=super-init-not-called
        """
        Here we calculate self-information for each item

        :param log: historical data
        """
        self._use_scala_udf = use_scala_udf
        self.log = convert2spark(log)
        n_users = self.log.select("user_idx").distinct().count()  # type: ignore
        self.item_weights = self.log.groupby("item_idx").agg(
            (
                sf.log2(n_users / sf.countDistinct("user_idx"))  # type: ignore
                / np.log2(n_users)
            ).alias("rec_weight")
        )

    @staticmethod
    def _get_metric_value_by_user(k, *args):
        weigths = args[0]
        return sum(weigths[:k]) / k

    @staticmethod
    def _get_metric_value_by_user_scala_udf(
            k: Union[str, Column],
            weigths: Union[str, Column]
    ) -> Column:
        return RecOnlyMetric.get_scala_udf('getSurprisalMetricValue', [k, weigths])

    def _get_enriched_recommendations(
        self,
        recommendations: DataFrame,
        ground_truth: DataFrame,
        max_k: int,
        ground_truth_users: Optional[AnyDataFrame] = None,
    ) -> DataFrame:
        recommendations = convert2spark(recommendations)
        ground_truth_users = convert2spark(ground_truth_users)
        recommendations = get_top_k_recs(recommendations, max_k)
        sort_udf = sf.udf(
            partial(sorter, extra_position=2),
            returnType=st.StructType(
                [
                    st.StructField(
                        "pred",
                        st.ArrayType(
                            self.item_weights.schema["item_idx"].dataType
                        ),
                    ),
                    st.StructField(
                        "rec_weight",
                        st.ArrayType(
                            self.item_weights.schema["rec_weight"].dataType
                        ),
                    ),
                ],
            ),
        )
        recommendations = (
            recommendations.join(self.item_weights, on="item_idx", how="left")
            .fillna(1.0)
            .groupby("user_idx")
            .agg(
                sf.collect_list(
                    sf.struct("relevance", "item_idx", "rec_weight")
                ).alias("rel_id_weight")
            )
            .withColumn("pred_rec_weight", sort_udf(sf.col("rel_id_weight")))
            .select(
                "user_idx",
                sf.col("pred_rec_weight.rec_weight").alias("rec_weight"),
            )
        )

        if ground_truth_users is not None:
            recommendations = fill_na_with_empty_array(
                recommendations.join(
                    ground_truth_users, on="user_idx", how="right"
                ),
                "rec_weight",
                self.item_weights.schema["rec_weight"].dataType,
            )

        return recommendations
