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
"""

    def _get_metric_value_by_user(self, pred, ground_truth, k) -> float:
        for i in pred[:k]:
            if i in ground_truth:
                return 1.0
        return 0.0
