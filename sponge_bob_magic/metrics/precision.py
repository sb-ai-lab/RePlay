"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from sponge_bob_magic.metrics.base_metric import Metric


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
    def _get_metric_value_by_user(pandas_df):
        pandas_df = pandas_df.assign(
            is_good_item=pandas_df[["item_id", "items_id"]].apply(
                lambda x: int(x["item_id"] in x["items_id"]), 1
            )
        )
        return pandas_df.assign(
            cum_agg=pandas_df["is_good_item"].cumsum() / pandas_df.k
        )
