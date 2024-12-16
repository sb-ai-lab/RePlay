from typing import List

from .base_metric import Metric


class HitRate(Metric):
    """
    Percentage of users that have at least one correctly recommended item\
        among top-k.

    .. math::
        HitRate@K(i) = \\max_{j \\in [1..K]}\\mathbb{1}_{r_{ij}}

    .. math::
        HitRate@K = \\frac {\\sum_{i=1}^{N}HitRate@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function stating that user :math:`i` interacted with item :math:`j`

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
    >>> HitRate(2)(recommendations, groundtruth)
    {'HitRate@2': 0.6666666666666666}
    >>> HitRate(2, mode=PerUser())(recommendations, groundtruth)
    {'HitRate-PerUser@2': {1: 1.0, 2: 0.0, 3: 1.0}}
    >>> HitRate(2, mode=Median())(recommendations, groundtruth)
    {'HitRate-Median@2': 1.0}
    >>> HitRate(2, mode=ConfidenceInterval(alpha=0.95))(recommendations, groundtruth)
    {'HitRate-ConfidenceInterval@2': 0.6533213281800181}
    <BLANKLINE>
    """

    @staticmethod
    def _get_metric_value_by_user(ks: List[int], ground_truth: List, pred: List) -> List[float]:
        if not ground_truth or not pred:
            return [0.0 for _ in ks]
        set_gt = set(ground_truth)
        res = []
        for k in ks:
            if len(set(pred[:k]).intersection(set_gt)) > 0:
                res.append(1.0)
            else:
                res.append(0.0)
        return res
