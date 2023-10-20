from replay.experimental.metrics.base_metric import ScalaMetric


# pylint: disable=too-few-public-methods
class ScalaHitRate(ScalaMetric):
    """
    Percentage of users that have at least one
        correctly recommended item among top-k.

    .. math::
        HitRate@K(i) = \max_{j \in [1..K]}\mathbb{1}_{r_{ij}}

    .. math::
        HitRate@K = \\frac {\sum_{i=1}^{N}HitRate@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function stating that user :math:`i` interacted with item :math:`j`

    """

    _scala_udf_name = "getHitRateMetricValue"
