from typing import List

from .base_metric import Metric


# pylint: disable=too-few-public-methods
class Recall(Metric):
    """
    Recall measures the coverage of the recommended items, and is defined as:

    Mean percentage of relevant items, that was shown among top ``K`` recommendations.

    .. math::
        Recall@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij}}}{|Rel_i|}

    .. math::
        Recall@K = \\frac {\sum_{i=1}^{N}Recall@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing that user :math:`i` interacted with item :math:`j`

    :math:`|Rel_i|` -- the number of relevant items for user :math:`i`

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
    >>> Recall(2)(recommendations, groundtruth)
    {'Recall@2': 0.12222222222222223}
    >>> Recall(2, mode=PerUser())(recommendations, groundtruth)
    {'Recall-PerUser@2': {1: 0.16666666666666666, 2: 0.0, 3: 0.2}}
    >>> Recall(2, mode=Median())(recommendations, groundtruth)
    {'Recall-Median@2': 0.16666666666666666}
    >>> Recall(2, mode=ConfidenceInterval(alpha=0.95))(recommendations, groundtruth)
    {'Recall-ConfidenceInterval@2': 0.12125130695058273}
    <BLANKLINE>
    """

    @staticmethod
    def _get_metric_value_by_user(  # pylint: disable=arguments-differ
        ks: List[int], ground_truth: List, pred: List
    ) -> List[float]:  # pragma: no cover
        if not ground_truth or not pred:
            return [0.0 for _ in ks]
        set_gt = set(ground_truth)
        res = []
        for k in ks:
            ans = len(set(pred[:k]) & set(ground_truth))
            res.append(ans / len(set_gt))
        return res
