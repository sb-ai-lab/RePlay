from replay.metrics.base_metric import Metric

from pyspark.sql import SparkSession, Column
from pyspark.sql.column import _to_java_column, _to_seq


# pylint: disable=too-few-public-methods
class MAP(Metric):
    """
    Mean Average Precision -- average the ``Precision`` at relevant positions for each user,
        and then calculate the mean across all users.

    .. math::
        &AP@K(i) = \\frac 1K \sum_{j=1}^{K}\mathbb{1}_{r_{ij}}Precision@j(i)

        &MAP@K = \\frac {\sum_{i=1}^{N}AP@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing if user :math:`i` interacted with item :math:`j`
    """

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        length = min(k, len(pred))
        max_good = min(k, len(ground_truth))
        if len(ground_truth) == 0 or len(pred) == 0:
            return 0
        tp_cum = 0
        result = 0
        for i in range(length):
            if pred[i] in ground_truth:
                tp_cum += 1
                result += tp_cum / ((i + 1) * max_good)
        return result

    @staticmethod
    def _get_metric_value_by_user_scala_udf(k, pred, ground_truth) -> Column:
        sc = SparkSession.getActiveSession().sparkContext
        _f = (
            sc._jvm.org.apache.spark.replay.utils.ScalaPySparkUDFs.getMAPMetricValue()
        )
        return Column(
            _f.apply(_to_seq(sc, [k, pred, ground_truth], _to_java_column))
        )
