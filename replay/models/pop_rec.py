from typing import Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import Recommender
from replay.constants import IDX_REC_SCHEMA


class PopRec(Recommender):
    """
    Recommend objects using their popularity.

    Popularity of an item is a probability that random user rated this item.

    .. math::
        Popularity(i) = \\dfrac{N_i}{N}

    :math:`N_i` - number of users who rated item :math:`i`

    :math:`N` - total number of users

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_id": [1, 1, 2, 2, 3, 4], "item_id": [1, 2, 2, 3, 3, 3]})
    >>> data_frame
       user_id  item_id
    0        1        1
    1        1        2
    2        2        2
    3        2        3
    4        3        3
    5        4        3

    >>> from replay.utils import convert2spark
    >>> res = PopRec().fit_predict(convert2spark(data_frame), 1)
    >>> res.toPandas().sort_values("user_id", ignore_index=True)
       user_id  item_id  relevance
    0        1        3       0.75
    1        2        1       0.25
    2        3        2       0.50
    3        4        2       0.50

    >>> res = PopRec().fit_predict(convert2spark(data_frame), 1, filter_seen_items=False)
    >>> res.toPandas().sort_values("user_id", ignore_index=True)
       user_id  item_id  relevance
    0        1        3       0.75
    1        2        3       0.75
    2        3        3       0.75
    3        4        3       0.75
    """

    item_popularity: DataFrame
    can_predict_cold_users = True

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.item_popularity = (
            log.groupBy("item_idx")
            .agg(sf.countDistinct("user_idx").alias("user_count"))
            .select(
                "item_idx",
                (sf.col("user_count") / sf.lit(self.users_count)).alias(
                    "relevance"
                ),
            )
        )
        self.item_popularity.cache()

    def _clear_cache(self):
        if hasattr(self, "item_popularity"):
            self.item_popularity.unpersist()

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
        items_pd = (
            items.join(
                self.item_popularity.withColumnRenamed("item_idx", "item"),
                on=sf.col("item_idx") == sf.col("item"),
                how="inner",
            )
            .drop("item")
            .toPandas()
        )

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            user_idx = pandas_df["user_idx"][0]
            cnt = pandas_df["cnt"][0]

            items_idx = np.argsort(items_pd["relevance"].values)[-cnt:]

            return pd.DataFrame(
                {
                    "user_idx": cnt * [user_idx],
                    "item_idx": items_pd["item_idx"].values[items_idx],
                    "relevance": items_pd["relevance"].values[items_idx],
                }
            )

        model_len = len(items_pd)
        recs = (
            users.join(log, how="left", on="user_idx")
            .select("user_idx", "item_idx")
            .groupby("user_idx")
            .agg(sf.countDistinct("item_idx").alias("cnt"))
        )
        recs = (
            recs.selectExpr(
                "user_idx",
                f"LEAST(cnt + {k}, {model_len}) AS cnt",
            )
            .groupby("user_idx")
            .applyInPandas(grouped_map, IDX_REC_SCHEMA)
        )

        return recs

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        return pairs.join(self.item_popularity, on="item_idx", how="inner")
