import numpy as np
import pandas as pd

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

    :math:`|Rel_i|` -- количество элементов, с которыми пользователь :math:`i` взаимодействовал

    Для расчета итоговой метрики усредняем по всем пользователям

    .. math::
        nDCG@K = \\frac {\sum_{i=1}^{N}nDCG@K(i)}{N}
    """

    @staticmethod
    def _get_metric_value_by_user(pandas_df: pd.DataFrame) -> pd.DataFrame:
        def check_is_in(dataframe: pd.DataFrame):
            return int(dataframe["item_id"] in dataframe["items_id"])

        pandas_df = pandas_df.assign(  # type: ignore
            is_good_item=pandas_df[["item_id", "items_id"]].apply(
                check_is_in, 1
            )
        )
        pandas_df = pandas_df.assign(  # type: ignore
            sorted_good_item=pandas_df["k"].le(  # type: ignore
                pandas_df["items_id"].str.len()
            )  # type: ignore
        )
        return pandas_df.assign(  # type: ignore
            cum_agg=(
                pandas_df["is_good_item"]
                / np.log2(pandas_df["k"] + 1)  # type: ignore
            ).cumsum()
            / (
                pandas_df["sorted_good_item"]
                / np.log2(pandas_df["k"] + 1)  # type: ignore
            ).cumsum()
        )
