"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from replay.metrics.base_metric import Metric


# pylint: disable=too-few-public-methods
class HitRate(Metric):
    """
    Доля пользователей, для которой хотя бы одна рекомендация из
    первых ``K`` была успешна.

    .. math::
        HitRate@K(i) = \max_{j \in [1..K]}\mathbb{1}_{r_{ij}}

    .. math::
        HitRate@K = \\frac {\sum_{i=1}^{N}HitRate@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- индикатор взаимодействия пользователя :math:`i` с рекомендацией :math:`j`

    >>> hr = HitRate()
    >>> hr._get_metric_value_by_user(4, [1,2,3,4], [2,4])
    1.0

    >>> hr._get_metric_value_by_user(4, [1,2,3,4], [5,6])
    0.0
    """

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        for i in pred[:k]:
            if i in ground_truth:
                return 1.0
        return 0.0
