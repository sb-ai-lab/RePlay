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

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        for i in pred[:k]:
            if i in ground_truth:
                return 1
        return 0
