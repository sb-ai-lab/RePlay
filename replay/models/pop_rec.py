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
        selected_item_popularity = self.item_popularity.join(
            items,
            on="item_idx",
            how="inner",
        ).withColumn(
            "rank",
            sf.row_number().over(Window.orderBy(sf.col("relevance").desc())),
        )

        if not filter_seen_items:
            return users.crossJoin(
                selected_item_popularity.filter(sf.col("rank") <= k)
            ).drop("rank")

        log_by_user = (
            log.join(users, on="user_idx")
            .groupBy("user_idx")
            .agg(sf.countDistinct("item_idx").alias("items_count"))
        )
        max_history_len = log_by_user.select(sf.max("items_count")).collect()[
            0
        ][0]
        cropped_item_popularity = selected_item_popularity.filter(
            sf.col("rank") <= max_history_len + k
        )

        log_by_user_with_new_users = log_by_user.join(
            users, on="user_idx", how="right"
        ).fillna(0)
        recs = log_by_user_with_new_users.join(
            cropped_item_popularity,
            on=sf.col("rank") <= sf.col("items_count") + sf.lit(k),
        ).drop("rank", "items_count")

        return recs

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        return pairs.join(self.item_popularity, on="item_idx", how="inner")
