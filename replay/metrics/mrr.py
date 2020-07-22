"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from replay.metrics.base_metric import Metric


# pylint: disable=too-few-public-methods
class MRR(Metric):
    """
    Mean Reciprocal Rank --
    для списка рекомендаций i, Reciprocal Rank определяется как позиция первой релевантной рекомендации
    в минус первой степени, то есть  :math:`\\frac {1}{rank_i}`. Это значение усредняется по всем пользователям.

    >>> import pandas as pd
    >>> pred = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [3, 2, 1], "relevance": [5 ,5, 5]})
    >>> true = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [2, 4, 5], "relevance": [5, 5, 5]})
    >>> MRR()(pred, true, 3)
    0.5
    >>> MRR()(true, pred, 1)
    1.0
    """

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        pandas_df = pandas_df.assign(
            is_good_item=pandas_df[["item_id", "items_id", "k"]].apply(
                lambda x: int(x["item_id"] in x["items_id"]) / x["k"], 1
            )
        )
        return pandas_df.assign(cum_agg=pandas_df.is_good_item.cummax())
