from typing import Optional

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf

from replay.models.base_rec import Recommender


class PopRec(Recommender):
    """
    Recommend objects using their popularity.

    Popularity of an item is a probability that random user rated this item.

    .. math::
        Popularity(i) = \\dfrac{N_i}{N}

    :math:`N_i` - number of users who rated item :math:`i`

    :math:`N` - total number of users

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 1, 2, 2, 3, 4], "item_idx": [1, 2, 2, 3, 3, 3], "relevance": [0.5, 1, 0.1, 0.8, 0.7, 1]})
    >>> data_frame
       user_idx  item_idx  relevance
    0         1         1        0.5
    1         1         2        1.0
    2         2         2        0.1
    3         2         3        0.8
    4         3         3        0.7
    5         4         3        1.0

    >>> from replay.utils import convert2spark
    >>> data_frame = convert2spark(data_frame)

    >>> res = PopRec().fit_predict(data_frame, 1)
    >>> res.toPandas().sort_values("user_idx", ignore_index=True)
       user_idx  item_idx  relevance
    0         1         3       0.75
    1         2         1       0.25
    2         3         2       0.50
    3         4         2       0.50

    >>> res = PopRec().fit_predict(data_frame, 1, filter_seen_items=False)
    >>> res.toPandas().sort_values("user_idx", ignore_index=True)
       user_idx  item_idx  relevance
    0         1         3       0.75
    1         2         3       0.75
    2         3         3       0.75
    3         4         3       0.75

    >>> res = PopRec(use_relevance=True).fit_predict(data_frame, 1)
    >>> res.toPandas().sort_values("user_idx", ignore_index=True)
       user_idx  item_idx  relevance
    0         1         3      0.625
    1         2         1      0.125
    2         3         2      0.275
    3         4         2      0.275

    """

    item_popularity: DataFrame
    can_predict_cold_users = True

    def __init__(self, use_relevance: bool = False):
        """
        :param use_relevance: flag to use relevance values as is or to treat them as 1
        """
        self.use_relevance = use_relevance

    @property
    def _init_args(self):
        return {"use_relevance": self.use_relevance}

    @property
    def _dataframes(self):
        return {"item_popularity": self.item_popularity}

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        if self.use_relevance:
            self.item_popularity = (
                log.groupBy("item_idx")
                .agg(sf.sum("relevance").alias("relevance"))
                .withColumn(
                    "relevance", sf.col("relevance") / sf.lit(self.users_count)
                )
            )
        else:
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
        selected_item_popularity = self.item_popularity.join(
            items,
            on="item_idx",
            how="inner",
        ).withColumn(
            "rank",
            sf.row_number().over(Window.orderBy(sf.col("relevance").desc())),
        )

        max_hist_len = 0
        if filter_seen_items:
            max_hist_len = (
                (
                    log.join(users, on="user_idx")
                    .groupBy("user_idx")
                    .agg(sf.countDistinct("item_idx").alias("items_count"))
                )
                .select(sf.max("items_count"))
                .collect()[0][0]
            )
            # all users have empty history
            if max_hist_len is None:
                max_hist_len = 0

        return users.crossJoin(
            selected_item_popularity.filter(sf.col("rank") <= k + max_hist_len)
        ).drop("rank")

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        return pairs.join(self.item_popularity, on="item_idx", how="inner")
