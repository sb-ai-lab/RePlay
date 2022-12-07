from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from replay.constants import AnyDataFrame
from replay.utils import convert2spark, get_top_k_recs
from replay.metrics.base_metric import RecOnlyMetric, sorter

from pyspark.sql import SparkSession, Column
from pyspark.sql.column import _to_java_column, _to_seq


# pylint: disable=too-few-public-methods
class Unexpectedness(RecOnlyMetric):
    """
    Fraction of recommended items that are not present in some baseline recommendations.

    >>> import pandas as pd
    >>> from replay.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> state = State(spark)

    >>> log = pd.DataFrame({"user_idx": [1, 1, 1], "item_idx": [1, 2, 3], "relevance": [5, 5, 5], "timestamp": [1, 1, 1]})
    >>> recs = pd.DataFrame({"user_idx": [1, 1, 1], "item_idx": [0, 0, 1], "relevance": [5, 5, 5], "timestamp": [1, 1, 1]})
    >>> metric = Unexpectedness(log)
    >>> round(metric(recs, 3), 2)
    0.67
    """

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

    @staticmethod
    def _get_metric_value_by_user_scala_udf(k, pred, base_pred) -> Column:
        sc = SparkSession.getActiveSession().sparkContext
        _f = (
            sc._jvm.org.apache.spark.replay.utils.ScalaPySparkUDFs.getUnexpectednessMetricValue()
        )
        return Column(
            _f.apply(_to_seq(sc, [k, pred, base_pred], _to_java_column))
        )

    def _get_enriched_recommendations(
        self, recommendations: DataFrame, ground_truth: DataFrame, max_k: int
    ) -> DataFrame:
        recommendations = convert2spark(recommendations)
        base_pred = self.pred
        sort_udf = sf.udf(
            sorter,
            returnType=st.ArrayType(base_pred.schema["item_idx"].dataType),
        )
        base_recs = (
            base_pred.groupby("user_idx")
            .agg(
                sf.collect_list(sf.struct("relevance", "item_idx")).alias(
                    "base_pred"
                )
            )
            .select(
                "user_idx", sort_udf(sf.col("base_pred")).alias("base_pred")
            )
        )
        recommendations = get_top_k_recs(recommendations, k=max_k)
        recommendations = (
            recommendations.groupby("user_idx")
            .agg(
                sf.collect_list(sf.struct("relevance", "item_idx")).alias(
                    "pred"
                )
            )
            .select("user_idx", sort_udf(sf.col("pred")).alias("pred"))
            .join(base_recs, how="right", on=["user_idx"])
        )
        return recommendations.withColumn(
            "pred",
            sf.coalesce(
                "pred",
                sf.array().cast(
                    st.ArrayType(base_pred.schema["item_idx"].dataType)
                ),
            ),
        )
