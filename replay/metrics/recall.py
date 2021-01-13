"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from replay.metrics.base_metric import Metric


# pylint: disable=too-few-public-methods
class Recall(Metric):
    """
    Какая доля объектов, с которыми взаимодействовал пользователь в тестовых данных, была показана ему в списке рекомендаций длины ``K``?

    .. math::
        Recall@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij}}}{|Rel_i|}

    .. math::
        Recall@K = \\frac {\sum_{i=1}^{N}Recall@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- индикатор взаимодействия пользователя :math:`i` с рекомендацией :math:`j`

    :math:`|Rel_i|` -- количество элементов, с которыми взаимодействовал пользователь :math:`i`
    """

    def _get_metric_value_by_user(self, k, pred, ground_truth) -> float:
        if len(ground_truth) == 0:
            return 0
        return len(set(pred[:k]) & set(ground_truth)) / len(ground_truth)
