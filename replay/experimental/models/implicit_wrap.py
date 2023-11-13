from os.path import join
from typing import Optional

import pandas as pd
from pyspark.sql import DataFrame

from replay.preprocessing import CSRConverter
from replay.experimental.models.base_rec import Recommender
from replay.utils.spark_utils import save_picklable_to_parquet, load_pickled_from_parquet
from replay.data import get_schema


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
    >>> from replay.utils.spark_utils import convert2spark
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
        save_picklable_to_parquet(self.model, join(path, "model"))

    def _load_model(self, path: str):
        self.model = load_pickled_from_parquet(join(path, "model"))

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        matrix = CSRConverter(
            first_dim_column="user_idx",
            second_dim_column="item_idx",
            data_column="relevance"
        ).transform(log)
        self.model.fit(matrix)

    @staticmethod
    def _pd_func(model, items_to_use=None, user_item_data=None, filter_seen_items=False):
        def predict_by_user_item(pandas_df):
            user = int(pandas_df["user_idx"].iloc[0])
            items = items_to_use if items_to_use else pandas_df.item_idx.to_list()

            items_res, rel = model.recommend(
                userid=user,
                user_items=user_item_data[user] if filter_seen_items else None,
                N=len(items),
                filter_already_liked_items=filter_seen_items,
                items=items,
            )
            return pd.DataFrame(
                {
                    "user_idx": [user] * len(items_res),
                    "item_idx": items_res,
                    "relevance": rel,
                }
            )

        return predict_by_user_item

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

        items_to_use = items.distinct().toPandas().item_idx.tolist()
        user_item_data = CSRConverter(
            first_dim_column="user_idx",
            second_dim_column="item_idx",
            data_column="relevance"
        ).transform(log)
        model = self.model
        rec_schema = get_schema(
            query_column="user_idx",
            item_column="item_idx",
            rating_column="relevance",
            has_timestamp=False,
        )
        return (
            users.select("user_idx")
            .groupby("user_idx")
            .applyInPandas(self._pd_func(
                model=model,
                items_to_use=items_to_use,
                user_item_data=user_item_data,
                filter_seen_items=filter_seen_items), rec_schema)
        )

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:

        model = self.model
        rec_schema = get_schema(
            query_column="user_idx",
            item_column="item_idx",
            rating_column="relevance",
            has_timestamp=False,
        )
        return pairs.groupby("user_idx").applyInPandas(
            self._pd_func(model=model, filter_seen_items=False),
            rec_schema)
