from replay.metrics.base_metric import Metric, get_enriched_recommendations


# pylint: disable=too-few-public-methods
class MRR(Metric):
    """
    Mean Reciprocal Rank --
    Reciprocal Rank определяется как обратная позиция первой релевантной рекомендации (i) в списке первых K
    рекомендаций, то есть  :math:`\\frac {1}{rank_i}`. Это значение усредняется по всем пользователям.

    >>> import pandas as pd
    >>> pred = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [3, 2, 1], "relevance": [5 ,5, 5]})
    >>> true = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [2, 4, 5], "relevance": [5, 5, 5]})
    >>> MRR()(pred, true, 3)
    0.5
    >>> MRR()(pred, true, 1)
    0.0
    >>> MRR()(true, pred, 1)
    1.0
    """

    def __init__(self):
        self._get_enriched_recommendations = get_enriched_recommendations

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        for i in range(min(k, len(pred))):
            if pred[i] in ground_truth:
                return 1 / (1 + i)
        return 0
