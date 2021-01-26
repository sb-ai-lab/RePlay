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

    Для перехода от :math:`DCG` к :math:`nDCG` необходимо подсчитать максимальное значение метрики
    для пользователя :math:`i` и  длины рекомендаций :math:`K`

    .. math::
        IDCG@K(i) = max(DCG@K(i)) = \sum_{j=1}^{K}\\frac{\mathbb{1}_{j\le|Rel_i|}}{\log_2 (j+1)}

    .. math::
        nDCG@K(i) = \\frac {DCG@K(i)}{IDCG@K(i)}

    :math:`|Rel_i|` -- количество элементов, с которыми пользователь :math:`i` взаимодействовал

    Для расчета итоговой метрики усредняем по всем пользователям

    .. math::
        nDCG@K = \\frac {\sum_{i=1}^{N}nDCG@K(i)}{N}

    >>> import pandas as pd
    >>> pred=pd.DataFrame({"user_id": [1, 1, 2, 2],
    ...                    "item_id": [4, 5, 6, 7],
    ...                    "relevance": [1, 1, 1, 1]})
    >>> true=pd.DataFrame({"user_id": [1, 1, 1, 1, 1, 2],
    ...                    "item_id": [1, 2, 3, 4, 5, 8],
    ...                    "relevance": [0.5, 0.1, 0.25, 0.6, 0.2, 0.3]})
    >>> ndcg = NDCG()
    >>> ndcg(pred, true, 2)
    0.5
    """

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        pred_len = min(k, len(pred))
        ground_truth_len = min(k, len(ground_truth))
        denom = [1 / math.log2(i + 2) for i in range(k)]
        dcg = sum(
            [denom[i] for i in range(pred_len) if pred[i] in ground_truth]
        )
        idcg = sum(denom[:ground_truth_len])

        return dcg / idcg
