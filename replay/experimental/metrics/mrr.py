from replay.experimental.metrics.base_metric import ScalaMetric


# pylint: disable=too-few-public-methods
class ScalaMRR(ScalaMetric):
    """
    Mean Reciprocal Rank --
    Reciprocal Rank is the inverse position of the first relevant item among top-k recommendations,
    :math:`\\frac {1}{rank_i}`. This value is averaged by all users.
    """

    _scala_udf_name = "getMRRMetricValue"
