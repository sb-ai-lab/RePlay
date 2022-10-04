from typing import Optional

import joblib
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame

from replay.models.base_rec import Recommender
from replay.utils import to_csr, convert2spark
from replay.constants import REC_SCHEMA


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
    >>> from replay.utils import convert2spark
    >>> df = pd.DataFrame({"user_idx": [1, 1, 2, 2], "item_idx": [1, 2, 2, 3], "relevance": [1, 1, 1, 1]})
    >>> df = convert2spark(df)
    >>> als.fit_predict(df, 1, users=[1])[["user_idx", "item_idx"]].toPandas()
       user_idx  item_idx
    0         1         3
    """

    def __init__(self, model):
        """Provide initialized ``implicit`` model."""
        self.model = model
        self.logger.info(
            "The model is a wrapper of a non-distributed model which may affect performance"
        )

    @property
    def _init_args(self):
        return {"model": None}

    def _save_model(self, path: str):
        joblib.dump(self.model, path)

    def _load_model(self, path: str):
        self.model = joblib.load(path)

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        matrix = to_csr(log)
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
            ids, rel = model.recommend(
                userid=user,
                user_items=user_item_data[user],
                N=k,
                filter_already_liked_items=filter_seen_items,
                items=items_to_use
            )
            return pd.DataFrame(
                {
                    "user_idx": [user] * len(ids),
                    "item_idx": ids,
                    "relevance": rel,
                }
            )
        items_to_use = items.distinct().toPandas().item_idx.tolist()
        user_item_data = to_csr(log)
        model = self.model
        return (
            users.select("user_idx")
            .groupby("user_idx")
            .applyInPandas(predict_by_user, REC_SCHEMA)
        )

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:

        def predict_by_user_item(pandas_df: pd.DataFrame) -> pd.DataFrame:
            user = int(pandas_df["user_idx"].iloc[0])
            items = pandas_df.item_idx.to_list()
            item_grid, rel = model.recommend(
                userid=user,
                user_items=user_item_data[user] if user_item_data is not None else None,
                N=len(items),
                filter_already_liked_items=False,
                items=items,
            )
            return pd.DataFrame(
                {
                    "user_idx": [user] * len(items),
                    "item_idx": item_grid,
                    "relevance": rel,
                }
            )
        user_item_data = to_csr(log) if log is not None else None
        model = self.model
        return pairs.groupby("user_idx").applyInPandas(predict_by_user_item, REC_SCHEMA)
