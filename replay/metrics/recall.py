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

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        pandas_df = pandas_df.assign(
            is_good_item=pandas_df[["item_id", "items_id"]].apply(
                lambda x: int(x["item_id"] in x["items_id"]), 1
            )
        )
        return pandas_df.assign(
            cum_agg=pandas_df["is_good_item"].cumsum()
            / pandas_df["items_id"].str.len()
        )
