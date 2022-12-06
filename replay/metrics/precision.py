from replay.metrics.base_metric import Metric

from pyspark.sql import SparkSession, Column
from pyspark.sql.column import _to_java_column, _to_seq


# pylint: disable=too-few-public-methods
class Precision(Metric):
    """
    Mean percentage of relevant items among top ``K`` recommendations.

    .. math::
        Precision@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij}}}{K}

    .. math::
        Precision@K = \\frac {\sum_{i=1}^{N}Precision@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing that user :math:`i` interacted with item :math:`j`"""

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        if len(pred) == 0:
            return 0
        return len(set(pred[:k]) & set(ground_truth)) / len(pred[:k])

    @staticmethod
    def _get_metric_value_by_user_scala_udf(k, pred, ground_truth) -> Column:
        sc = SparkSession.getActiveSession().sparkContext
        _f = (
            sc._jvm.org.apache.spark.replay.utils.ScalaPySparkUDFs.getPrecisionMetricValue()
        )
        return Column(
            _f.apply(_to_seq(sc, [k, pred, ground_truth], _to_java_column))
        )
