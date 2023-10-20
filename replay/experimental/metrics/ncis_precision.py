from replay.experimental.metrics.base_metric import ScalaMetric
from replay.metrics.base_metric import NCISMetric


# pylint: disable=too-few-public-methods
class ScalaNCISPrecision(NCISMetric, ScalaMetric):
    """
    Share of relevant items among top ``K`` recommendations with NCIS weighting.

    .. math::
        Precision@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij} w_{ij}}}{\sum_{j=1}^{K} w_{ij}}

    .. math::
        Precision@K = \\frac {\sum_{i=1}^{N}Precision@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function
        showing that user :math:`i` interacted with item :math:`j`
    :math:`w_{ij}` -- NCIS weight, calculated as ratio of current policy score on previous
        policy score with clipping and optional activation over policy scores (relevance).
        Source: arxiv.org/abs/1801.07030
    """

    _scala_udf_name = "getNCISPrecisionMetricValue"
