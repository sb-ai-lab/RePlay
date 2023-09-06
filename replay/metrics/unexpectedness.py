from typing import Optional

from pyspark.sql import DataFrame

from replay.data import AnyDataFrame
from replay.metrics.base_metric import (
    RecOnlyMetric,
    fill_na_with_empty_array,
    filter_sort
)
from replay.utils.spark_utils import convert2spark, get_top_k_recs


# pylint: disable=too-few-public-methods
class Unexpectedness(RecOnlyMetric):
    """
    Fraction of recommended items that are not present in some baseline recommendations.

    >>> import pandas as pd
    >>> from replay.utils.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> state = State(spark)

    >>> log = pd.DataFrame({"user_idx": [1, 1, 1], "item_idx": [1, 2, 3], "relevance": [5, 5, 5], "timestamp": [1, 1, 1]})
    >>> recs = pd.DataFrame({"user_idx": [1, 1, 1], "item_idx": [0, 0, 1], "relevance": [5, 5, 5], "timestamp": [1, 1, 1]})
    >>> metric = Unexpectedness(log)
    >>> round(metric(recs, 3), 2)
    0.67
    """

    _scala_udf_name = "getUnexpectednessMetricValue"

    def __init__(
        self, pred: AnyDataFrame,
        use_scala_udf: bool = False
    ):  # pylint: disable=super-init-not-called
        """
        :param pred: model predictions
        """
        self._use_scala_udf = use_scala_udf
        self.pred = convert2spark(pred)

    @staticmethod
    def _get_metric_value_by_user(k, *args) -> float:
        pred = args[0]
        base_pred = args[1]
        if len(pred) == 0:
            return 0
        return 1.0 - len(set(pred[:k]) & set(base_pred[:k])) / k

    def _get_enriched_recommendations(
        self,
        recommendations: DataFrame,
        ground_truth: DataFrame,
        max_k: int,
        ground_truth_users: Optional[AnyDataFrame] = None,
    ) -> DataFrame:
        recommendations = convert2spark(recommendations)
        ground_truth_users = convert2spark(ground_truth_users)
        base_pred = self.pred

        # TO DO: preprocess base_recs once in __init__

        base_recs = filter_sort(base_pred).withColumnRenamed("pred", "base_pred")

        # if there are duplicates in recommendations,
        # we will leave fewer than k recommendations after sort_udf
        recommendations = get_top_k_recs(recommendations, k=max_k)
        recommendations = filter_sort(recommendations)
        recommendations = recommendations.join(base_recs, how="right", on=["user_idx"])

        if ground_truth_users is not None:
            recommendations = recommendations.join(
                ground_truth_users, on="user_idx", how="right"
            )

        return fill_na_with_empty_array(
            recommendations, "pred", base_pred.schema["item_idx"].dataType
        )
