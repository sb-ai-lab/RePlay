from typing import List

from .base_metric import Metric


# pylint: disable=too-few-public-methods
class MAP(Metric):
    """
    Mean Average Precision -- average the ``Precision`` at relevant positions \
        for each user, and then calculate the mean across all users.

    .. math::
        &AP@K(i) = \\frac {1}{\min(K, |Rel_i|)} \sum_{j=1}^{K}\mathbb{1}_{r_{ij}}Precision@j(i)

        &MAP@K = \\frac {\sum_{i=1}^{N}AP@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing if user :math:`i` interacted with item :math:`j`

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
    >>> MAP(2)(recommendations, groundtruth)
    {'MAP@2': 0.25}
    >>> MAP(2, mode=PerUser())(recommendations, groundtruth)
    {'MAP-PerUser@2': {1: 0.25, 2: 0.0, 3: 0.5}}
    >>> MAP(2, mode=Median())(recommendations, groundtruth)
    {'MAP-Median@2': 0.25}
    >>> MAP(2, mode=ConfidenceInterval(alpha=0.95))(recommendations, groundtruth)
    {'MAP-ConfidenceInterval@2': 0.282896433519043}
    <BLANKLINE>
    """

    @staticmethod
    def _get_metric_value_by_user(  # pylint: disable=arguments-differ
        ks: List[int], ground_truth: List, pred: List
    ) -> List[float]:  # pragma: no cover
        if not ground_truth or not pred:
            return [0.0 for _ in ks]
        res = []
        for k in ks:
            length = min(k, len(pred))
            max_good = min(k, len(ground_truth))
            tp_cum: int = 0
            result: float = 0.0
            for i in range(length):
                if pred[i] in ground_truth:
                    tp_cum += 1
                    result += tp_cum / (i + 1)
            res.append(result / max_good)
        return res
