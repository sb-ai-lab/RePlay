from typing import List

from .base_metric import Metric


# pylint: disable=too-few-public-methods
class RocAuc(Metric):
    """
    Receiver Operating Characteristic/Area Under the Curve is the aggregated performance measure,
    that depends only on the order of recommended items.
    It can be interpreted as the fraction of object pairs (object of class 1, object of class 0)
    that were correctly ordered by the model.
    The bigger the value of AUC, the better the classification model.

    .. math::
        ROCAUC@K(i) = \\frac {\sum_{s=1}^{K}\sum_{t=1}^{K}
        \mathbb{1}_{r_{si}<r_{ti}}
        \mathbb{1}_{gt_{si}<gt_{ti}}}
        {\sum_{s=1}^{K}\sum_{t=1}^{K} \mathbb{1}_{gt_{si}<gt_{tj}}}

    :math:`\\mathbb{1}_{r_{si}<r_{ti}}` -- indicator function showing that recommendation score for
    user :math:`i` for item :math:`s` is bigger than for item :math:`t`

    :math:`\mathbb{1}_{gt_{si}<gt_{ti}}` --  indicator function showing that
    user :math:`i` values item :math:`s` more than item :math:`t`.

    Metric is averaged by all users.

    .. math::
        ROCAUC@K = \\frac {\sum_{i=1}^{N}ROCAUC@K(i)}{N}

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
    >>> RocAuc(2)(recommendations, groundtruth)
    {'RocAuc@2': 0.3333333333333333}
    >>> RocAuc(2, mode=PerUser())(recommendations, groundtruth)
    {'RocAuc-PerUser@2': {1: 0.0, 2: 0.0, 3: 1.0}}
    >>> RocAuc(2, mode=Median())(recommendations, groundtruth)
    {'RocAuc-Median@2': 0.0}
    >>> RocAuc(2, mode=ConfidenceInterval(alpha=0.95))(recommendations, groundtruth)
    {'RocAuc-ConfidenceInterval@2': 0.6533213281800181}
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
            length = min(k, len(pred))
            fp_cur = 0
            fp_cum = 0
            for item in pred[:length]:
                if item in set_gt:
                    fp_cum += fp_cur
                else:
                    fp_cur += 1
            if fp_cur == length:
                res.append(0.0)
            elif fp_cum == 0:
                res.append(1.0)
            else:
                res.append(1 - fp_cum / (fp_cur * (length - fp_cur)))
        return res
