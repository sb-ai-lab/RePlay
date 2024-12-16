import math

from .base_metric import Metric


class NDCG(Metric):
    """
    Normalized Discounted Cumulative Gain is a metric
    that takes into account positions of relevant items.

    This is the binary version, it takes into account
    whether the item was consumed or not, relevance value is ignored.

    .. math::
        DCG@K(i) = \\sum_{j=1}^{K}\\frac{\\mathbb{1}_{r_{ij}}}{\\log_2 (j+1)}


    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing that user :math:`i` interacted with item :math:`j`

    To get from :math:`DCG` to :math:`nDCG` we calculate the biggest possible value of `DCG`
    for user :math:`i` and recommendation length :math:`K`.

    .. math::
        IDCG@K(i) = max(DCG@K(i)) = \\sum_{j=1}^{K}\\frac{\\mathbb{1}_{j\\le|Rel_i|}}{\\log_2 (j+1)}

    .. math::
        nDCG@K(i) = \\frac {DCG@K(i)}{IDCG@K(i)}

    :math:`|Rel_i|` -- number of relevant items for user :math:`i`

    Metric is averaged by users.

    .. math::
        nDCG@K = \\frac {\\sum_{i=1}^{N}nDCG@K(i)}{N}
    """

    _scala_udf_name = "getNDCGMetricValue"

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        if len(pred) == 0 or len(ground_truth) == 0:
            return 0.0
        pred_len = min(k, len(pred))
        ground_truth_len = min(k, len(ground_truth))
        denom = [1 / math.log2(i + 2) for i in range(k)]
        dcg = sum(denom[i] for i in range(pred_len) if pred[i] in ground_truth)
        idcg = sum(denom[:ground_truth_len])

        return dcg / idcg
