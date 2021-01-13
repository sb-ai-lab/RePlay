"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from replay.metrics.base_metric import Metric


# pylint: disable=too-few-public-methods
class MAP(Metric):
    """
    Mean Average Precision -- усреднение ``Precision`` по целым числам от 1 до ``K``, усреднённое по пользователям.

    .. math::
        &AP@K(i) = \\frac 1K \sum_{j=1}^{K}\mathbb{1}_{r_{ij}}Precision@j(i)

        &MAP@K = \\frac {\sum_{i=1}^{N}AP@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- индикатор взаимодействия пользователя :math:`i` с рекомендацией :math:`j`
    """

    def _get_metric_value_by_user(self, k, pred, ground_truth) -> float:
        length = min(k, len(pred))
        max_good = min(k, len(ground_truth))
        if len(ground_truth) == 0 or len(pred) == 0:
            return 0
        tp_cum = 0
        result = 0
        for i in range(length):
            if pred[i] in ground_truth:
                tp_cum += 1
                result += tp_cum / ((i + 1) * max_good)
        return result
