from replay.experimental.metrics.base_metric import ScalaMetric


# pylint: disable=too-few-public-methods
class ScalaRecall(ScalaMetric):
    """
    Mean percentage of relevant items, that was shown among top ``K`` recommendations.

    .. math::
        Recall@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij}}}{|Rel_i|}

    .. math::
        Recall@K = \\frac {\sum_{i=1}^{N}Recall@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing that user :math:`i` interacted with item :math:`j`

    :math:`|Rel_i|` -- the number of relevant items for user :math:`i`
    """

    _scala_udf_name = "getRecallMetricValue"
