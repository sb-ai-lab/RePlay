from typing import List

from .base_metric import Metric


# pylint: disable=too-few-public-methods
class MRR(Metric):
    """
    Mean Reciprocal Rank -- Reciprocal Rank is the inverse position of the
    first relevant item among top-k recommendations,
    :math:`\\frac{1}{rank_i}`. This value is aggregated by all users.

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
    >>> MRR(2)(recommendations, groundtruth)
    {'MRR@2': 0.5}
    >>> MRR(2, mode=PerUser())(recommendations, groundtruth)
    {'MRR-PerUser@2': {1: 0.5, 2: 0.0, 3: 1.0}}
    >>> MRR(2, mode=Median())(recommendations, groundtruth)
    {'MRR-Median@2': 0.5}
    >>> MRR(2, mode=ConfidenceInterval(alpha=0.95))(recommendations, groundtruth)
    {'MRR-ConfidenceInterval@2': 0.565792867038086}
    <BLANKLINE>
    """

    @staticmethod
    def _get_metric_value_by_user(  # pylint: disable=arguments-differ
        ks: List[int], ground_truth: List, pred: List
    ) -> List[float]:
        if not ground_truth or not pred:
            return [0.0 for _ in ks]
        set_gt = set(ground_truth)
        res = []
        for k in ks:
            ans = 0.0
            for index, item_id in enumerate(pred[:k]):
                if item_id in set_gt:
                    ans = 1 / (index + 1)
                    break
            res.append(ans)
        return res
