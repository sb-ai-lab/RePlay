from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from replay.constants import AnyDataFrame
from replay.utils import convert2spark
from replay.metrics.base_metric import RecOnlyMetric, sorter


# pylint: disable=too-few-public-methods
class Unexpectedness(RecOnlyMetric):
    """
    Fraction of recommended items that are not present in some baseline recommendations.

    >>> import pandas as pd
    >>> from replay.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> state = State(spark)

    >>> log = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [1, 2, 3], "relevance": [5, 5, 5], "timestamp": [1, 1, 1]})
    >>> recs = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [0, 0, 1], "relevance": [5, 5, 5], "timestamp": [1, 1, 1]})
    >>> metric = Unexpectedness(log)
    >>> round(metric(recs, 3), 2)
    0.67
    """

    def __init__(
        self, pred: AnyDataFrame
    ):  # pylint: disable=super-init-not-called
        """
        :param pred: model predictions
        """
        self.pred = convert2spark(pred)

    @staticmethod
    def _get_metric_value_by_user(k, *args) -> float:
        pred = args[0]
        base_pred = args[1]
        if len(pred) == 0:
            return 0
        return 1.0 - len(set(pred[:k]) & set(base_pred[:k])) / k

    def _get_enriched_recommendations(
        self, recommendations: DataFrame, ground_truth: DataFrame
    ) -> DataFrame:
        recommendations = convert2spark(recommendations)
        base_pred = self.pred
        sort_udf = sf.udf(
            sorter,
            returnType=st.ArrayType(base_pred.schema["item_id"].dataType),
        )
        base_recs = (
            base_pred.groupby("user_id")
            .agg(
                sf.collect_list(sf.struct("relevance", "item_id")).alias(
                    "base_pred"
                )
            )
            .select(
                "user_id", sort_udf(sf.col("base_pred")).alias("base_pred")
            )
        )

        recommendations = (
            recommendations.groupby("user_id")
            .agg(
                sf.collect_list(sf.struct("relevance", "item_id")).alias(
                    "pred"
                )
            )
            .select("user_id", sort_udf(sf.col("pred")).alias("pred"))
            .join(base_recs, how="right", on=["user_id"])
        )
        return recommendations.withColumn(
            "pred",
            sf.coalesce(
                "pred",
                sf.array().cast(
                    st.ArrayType(base_pred.schema["item_id"].dataType)
                ),
            ),
        )
