from replay.metrics.base_metric import Metric

from pyspark.sql import SparkSession, Column
from pyspark.sql.column import _to_java_column, _to_seq


# pylint: disable=too-few-public-methods
class MRR(Metric):
    """
    Mean Reciprocal Rank --
    Reciprocal Rank is the inverse position of the first relevant item among top-k recommendations,
    :math:`\\frac {1}{rank_i}`. This value is averaged by all users.

    >>> import pandas as pd
    >>> pred = pd.DataFrame({"user_idx": [1, 1, 1], "item_idx": [3, 2, 1], "relevance": [5 ,5, 5]})
    >>> true = pd.DataFrame({"user_idx": [1, 1, 1], "item_idx": [2, 4, 5], "relevance": [5, 5, 5]})
    >>> MRR()(pred, true, 3)
    0.5
    >>> MRR()(pred, true, 1)
    0.0
    >>> MRR()(true, pred, 1)
    1.0
    """

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        for i in range(min(k, len(pred))):
            if pred[i] in ground_truth:
                return 1 / (1 + i)
        return 0

    @staticmethod
    def _get_metric_value_by_user_scala_udf(k, pred, ground_truth) -> Column:
        sc = SparkSession.getActiveSession().sparkContext
        _f = (
            sc._jvm.org.apache.spark.replay.utils.ScalaPySparkUDFs.getMRRMetricValue()
        )
        return Column(
            _f.apply(_to_seq(sc, [k, pred, ground_truth], _to_java_column))
        )
