from typing import Optional

import pandas as pd
from pyspark.sql import DataFrame

from replay.models.base_rec import Recommender
from replay.utils import to_csr
from replay.constants import IDX_REC_SCHEMA


class ImplicitWrap(Recommender):
    """Wrapper for `implicit
    <https://github.com/benfred/implicit>`_

    Example:

    >>> import implicit
    >>> model = implicit.als.AlternatingLeastSquares(factors=5)
    >>> als = ImplicitWrap(model)

    This way you can use implicit models as any other in replay
    with conversions made under the hood.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"user_id": [1, 1, 2, 2], "item_id": [1, 2, 2, 3], "relevance": [1, 1, 1, 1]})
    >>> als.fit_predict(df, 1, users=[1])[["user_id", "item_id"]].toPandas()
       user_id  item_id
    0        1        3
    """

    def __init__(self, model):
        """Provide initialized ``implicit`` model."""
        self.model = model

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        matrix = to_csr(log).T
        self.model.fit(matrix)

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        def predict_by_user(pandas_df: pd.DataFrame) -> pd.DataFrame:
            user = int(pandas_df["user_idx"].iloc[0])
            res = model.recommend(
                user, user_item_data, k, filter_seen_items, items_to_drop
            )
            return pd.DataFrame(
                {
                    "user_idx": [user] * len(res),
                    "item_idx": [val[0] for val in res],
                    "relevance": [val[1] for val in res],
                }
            )

        items_to_drop = (
            log.select("item_idx")
            .subtract(items)
            .select("item_idx")
            .toPandas()
            .item_idx.to_list()
        )
        user_item_data = to_csr(log).tocsr()
        model = self.model
        return (
            users.select("user_idx")
            .groupby("user_idx")
            .applyInPandas(predict_by_user, IDX_REC_SCHEMA)
        )
