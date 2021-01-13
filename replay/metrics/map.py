from replay.metrics.base_metric import Metric


# pylint: disable=too-few-public-methods
class MAP(Metric):
    """
    Mean Average Precision -- усреднение ``Precision`` по целым числам от 1 до ``K``, усреднённое по пользователям.

    .. math::
        &AP@K(i) = \\frac 1K \sum_{j=1}^{K}\mathbb{1}_{r_{ij}}Precision@j(i)

        &MAP@K = \\frac {\sum_{i=1}^{N}AP@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- индикатор взаимодействия пользователя :math:`i` с рекомендацией :math:`j`
    """

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        pandas_df = pandas_df.assign(
            is_good_item=pandas_df[["item_id", "items_id"]].apply(
                lambda x: int(x["item_id"] in x["items_id"]), 1
            ),
            good_items_count=pandas_df["items_id"].str.len(),
        )

        return pandas_df.assign(
            cum_agg=(
                pandas_df["is_good_item"].cumsum()
                * pandas_df["is_good_item"]
                / pandas_df.k
                / pandas_df[["k", "good_items_count"]].min(axis=1)
            ).cumsum()
        )
