from replay.metrics.base_metric import Metric

from pyspark.sql import SparkSession, Column
from pyspark.sql.column import _to_java_column, _to_seq


# pylint: disable=too-few-public-methods
class HitRate(Metric):
    """
    Percentage of users that have at least one
        correctly recommended item among top-k.

    .. math::
        HitRate@K(i) = \max_{j \in [1..K]}\mathbb{1}_{r_{ij}}

    .. math::
        HitRate@K = \\frac {\sum_{i=1}^{N}HitRate@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function stating that user :math:`i` interacted with item :math:`j`

    """

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        for i in pred[:k]:
            if i in ground_truth:
                return 1
        return 0

    @staticmethod
    def _get_metric_value_by_user_scala_udf(k, pred, ground_truth) -> Column:
        sc = SparkSession.getActiveSession().sparkContext
        _f = (
            sc._jvm.org.apache.spark.replay.utils.ScalaPySparkUDFs.getHitRateMetricValue()
        )
        return Column(
            _f.apply(_to_seq(sc, [k, pred, ground_truth], _to_java_column))
        )
