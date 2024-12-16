import math
from typing import List

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

    >>> recommendations
       query_id  item_id  rating
    0         1        3    0.6
    1         1        7    0.5
    2         1       10    0.4
    3         1       11    0.3
    4         1        2    0.2
    5         2        5    0.6
    6         2        8    0.5
    7         2       11    0.4
    8         2        1    0.3
    9         2        3    0.2
    10        3        4    1.0
    11        3        9    0.5
    12        3        2    0.1
    >>> groundtruth
       query_id  item_id
    0         1        5
    1         1        6
    2         1        7
    3         1        8
    4         1        9
    5         1       10
    6         2        6
    7         2        7
    8         2        4
    9         2       10
    10        2       11
    11        3        1
    12        3        2
    13        3        3
    14        3        4
    15        3        5
    >>> from replay.metrics import Median, ConfidenceInterval, PerUser
    >>> NDCG(2)(recommendations, groundtruth)
    {'NDCG@2': 0.3333333333333333}
    >>> NDCG(2, mode=PerUser())(recommendations, groundtruth)
    {'NDCG-PerUser@2': {1: 0.38685280723454163, 2: 0.0, 3: 0.6131471927654584}}
    >>> NDCG(2, mode=Median())(recommendations, groundtruth)
    {'NDCG-Median@2': 0.38685280723454163}
    >>> NDCG(2, mode=ConfidenceInterval(alpha=0.95))(recommendations, groundtruth)
    {'NDCG-ConfidenceInterval@2': 0.3508565839953337}
    <BLANKLINE>
    """

    @staticmethod
    def _get_metric_value_by_user(ks: List[int], ground_truth: List, pred: List) -> List[float]:
        if not pred or not ground_truth:
            return [0.0 for _ in ks]
        set_gt = set(ground_truth)
        res = []
        for k in ks:
            pred_len = min(k, len(pred))
            ground_truth_len = min(k, len(ground_truth))
            denom = [1 / math.log2(i + 2) for i in range(k)]
            dcg = sum(denom[i] for i in range(pred_len) if pred[i] in set_gt)
            idcg = sum(denom[:ground_truth_len])
            res.append(dcg / idcg)
        return res
