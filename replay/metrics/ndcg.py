"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import math

from replay.metrics.base_metric import Metric


# pylint: disable=too-few-public-methods
class NDCG(Metric):
    """
    Normalized Discounted Cumulative Gain учитывает порядок в списке рекомендаций --
    чем ближе к началу списка полезные рекомендации, тем больше значение метрики.

    Реализован бинарный вариант релевантности -- был объект или нет,
    не произвольная шкала полезности вроде оценок.

    Метрика определяется следующим образом:

    .. math::
        DCG@K(i) = \sum_{j=1}^{K}\\frac{\mathbb{1}_{r_{ij}}}{\log_2 (j+1)}


    :math:`\\mathbb{1}_{r_{ij}}` -- индикатор взаимодействия пользователя :math:`i` с рекомендацией :math:`j`

    Для перехода от :math:`DCG` к :math:`nDCG` необходимо подсчитать максимальное значение метрики для пользователя :math:`i` и  длины рекомендаций :math:`K`

    .. math::
        IDCG@K(i) = max(DCG@K(i)) = \sum_{j=1}^{K}\\frac{\mathbb{1}_{j\le|Rel_i|}}{\log_2 (j+1)}

    .. math::
        nDCG@K(i) = \\frac {DCG@K(i)}{IDCG@K(i)}

    :math:`|Rel_i|` -- количество элементов, с которыми пользовтель :math:`i` взаимодействовал

    Для расчета итоговой метрики усредняем по всем пользователям

    .. math::
        nDCG@K = \\frac {\sum_{i=1}^{N}nDCG@K(i)}{N}
    """

    _max_k = 1000
    _denom = [math.log2(i + 2) for i in range(_max_k)]

    def _get_metric_value_by_user(self, pred, ground_truth, k) -> float:
        length = max(k, min(len(ground_truth), len(pred)))
        if len(ground_truth) == 0:
            return 0
        dcg = 0
        dcg_ideal = 0
        for i in range(length):
            if i < len(pred) and pred[i] in ground_truth:
                dcg += self._denom[i]
            if i < len(ground_truth):
                dcg_ideal += self._denom[i]
        return dcg / dcg_ideal
