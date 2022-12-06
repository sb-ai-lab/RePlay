from replay.metrics.base_metric import Metric

from pyspark.sql import SparkSession, Column
from pyspark.sql.column import _to_java_column, _to_seq


# pylint: disable=too-few-public-methods
class Recall(Metric):
    """
    Mean percentage of relevant items, that was shown among top ``K`` recommendations.

    .. math::
        Recall@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij}}}{|Rel_i|}

    .. math::
        Recall@K = \\frac {\sum_{i=1}^{N}Recall@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing that user :math:`i` interacted with item :math:`j`

    :math:`|Rel_i|` -- the number of relevant items for user :math:`i`
    """

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        return len(set(pred[:k]) & set(ground_truth)) / len(ground_truth)

    @staticmethod
    def _get_metric_value_by_user_scala_udf(k, pred, ground_truth) -> Column:
        sc = SparkSession.getActiveSession().sparkContext
        _f = (
            sc._jvm.org.apache.spark.replay.utils.ScalaPySparkUDFs.getRecallMetricValue()
        )
        return Column(
            _f.apply(_to_seq(sc, [k, pred, ground_truth], _to_java_column))
        )
