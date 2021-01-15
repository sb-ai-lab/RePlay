"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from replay.metrics.base_metric import Metric


# pylint: disable=too-few-public-methods
class RocAuc(Metric):
    """
    Receiver Operating Characteristic/Area Under the Curve -- площадь под
    кривой ошибок. Является агрегированной характеристикой качества
    модели, не зависящей от соотношения цен ошибок, зависит только от
    порядка объектов. Если описывать смысл метрики, то ROC-AUC равен доле пар
    объектов вида (объект класса 1, объект класса 0), которые алгоритм верно
    упорядочил. Чем больше значение AUC, тем лучше модель классификации.

    Метрика определяется следующим образом:

    .. math::
        ROCAUC@K(i) = \\frac {\sum_{s=1}^{K}\sum_{t=1}^{K}
        \mathbb{1}_{r_{si}<r_{ti}}
        \mathbb{1}_{gt_{si}<gt_{ti}}}
        {\sum_{s=1}^{K}\sum_{t=1}^{K} \mathbb{1}_{gt_{si}<gt_{tj}}}

    :math:`\\mathbb{1}_{r_{si}<r_{ti}}` -- индикатор того, лучше ли для
    пользователя :math:`i` рекомендация :math:`s` в сравнении с рекомендацией :math:`t`

    :math:`\mathbb{1}_{gt_{si}<gt_{ti}}` --  индикатор того, лучше ли для
    пользователя :math:`i` было взаимодействие с объектом :math:`s` в
    сравнении с взаимодействием с объектом :math:`t`

    Для расчета итоговой метрики усредняем по всем пользователям

    .. math::
        ROCAUC@K = \\frac {\sum_{i=1}^{N}ROCAUC@K(i)}{N}

    >>> import pandas as pd
    >>> true=pd.DataFrame({"user_id": 1,
    ...                    "item_id": [4, 5, 6],
    ...                    "relevance": [1, 1, 1]})
    >>> pred=pd.DataFrame({"user_id": 1,
    ...                    "item_id": [1, 2, 3, 4, 5, 6, 7],
    ...                    "relevance": [0.5, 0.1, 0.25, 0.6, 0.2, 0.3, 0]})
    >>> roc = RocAuc()
    >>> roc(pred, true, 7)
    0.75

    >>> roc = RocAuc()
    >>> roc._get_metric_value_by_user(4, [1,2,3,4], [2,4])
    0.25

    >>> #Roc auc при полном промахе

    >>> roc = RocAuc()
    >>> roc._get_metric_value_by_user(4, [1,2,3,4], [5,6])
    0

    """

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        length = min(k, len(pred))
        if len(ground_truth) == 0 or len(pred) == 0:
            return 0

        fp_cur = 0
        fp_cum = 0
        for item in pred[:length]:
            if item in ground_truth:
                fp_cum += fp_cur
            else:
                fp_cur += 1
        if fp_cur == length:
            return 0
        if fp_cum == 0:
            return 1
        return 1 - fp_cum / (fp_cur * (length - fp_cur))
