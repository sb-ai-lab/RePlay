from replay.metrics.base_metric import Metric


# pylint: disable=too-few-public-methods
class Precision(Metric):
    """
    Средняя доля успешных рекомендаций среди первых ``K`` элементов выдачи.

    .. math::
        Precision@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij}}}{K}

    .. math::
        Precision@K = \\frac {\sum_{i=1}^{N}Precision@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- индикатор взаимодействия пользователя :math:`i` с рекомендацией :math:`j`
"""

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        if len(pred) == 0:
            return 0
        return len(set(pred[:k]) & set(ground_truth)) / len(pred[:k])
