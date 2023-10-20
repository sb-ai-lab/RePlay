from replay.experimental.metrics.base_metric import ScalaMetric


# pylint: disable=too-few-public-methods
class ScalaNDCG(ScalaMetric):
    """
    Normalized Discounted Cumulative Gain is a metric
    that takes into account positions of relevant items.

    This is the binary version, it takes into account
    whether the item was consumed or not, relevance value is ignored.

    .. math::
        DCG@K(i) = \sum_{j=1}^{K}\\frac{\mathbb{1}_{r_{ij}}}{\log_2 (j+1)}


    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing that user :math:`i` interacted with item :math:`j`

    To get from :math:`DCG` to :math:`nDCG` we calculate the biggest possible value of `DCG`
    for user :math:`i` and recommendation length :math:`K`.

    .. math::
        IDCG@K(i) = max(DCG@K(i)) = \sum_{j=1}^{K}\\frac{\mathbb{1}_{j\le|Rel_i|}}{\log_2 (j+1)}

    .. math::
        nDCG@K(i) = \\frac {DCG@K(i)}{IDCG@K(i)}

    :math:`|Rel_i|` -- number of relevant items for user :math:`i`

    Metric is averaged by users.

    .. math::
        nDCG@K = \\frac {\sum_{i=1}^{N}nDCG@K(i)}{N}
    """

    _scala_udf_name = "getNDCGMetricValue"
