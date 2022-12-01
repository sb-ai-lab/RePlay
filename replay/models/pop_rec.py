from typing import Optional, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import NonPersonalizedRecommender


class PopRec(NonPersonalizedRecommender):
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

    def __init__(self, use_relevance: bool = False):
        """
        :param use_relevance: flag to use relevance values as is or to treat them as 1
        """
        self.use_relevance = use_relevance

    @property
    def _init_args(self):
        return {"use_relevance": self.use_relevance}

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:

        if self.use_relevance:
            # we will save it to update fitted model
            self.item_abs_relevances = (
                log.groupBy("item_idx")
                .agg(sf.sum("relevance").alias("relevance"))
            )
            self._users_count = self.users_count
            self.all_user_ids = log.select("user_idx").distinct()

            self.item_popularity = (
                self.item_abs_relevances
                .withColumn(
                    "relevance", sf.col("relevance") / sf.lit(self._users_count)
                )
            )
        else:
            self.item_users = (
                log.groupBy("item_idx")
                .agg(sf.collect_set('user_idx').alias('user_idx'))
            )
            self._users_count = self.users_count
            self.all_user_ids = log.select("user_idx").distinct()

            self.item_popularity = (
                self.item_users
                .select(
                    "item_idx",
                    (sf.size("user_idx") / sf.lit(self.users_count)).alias(
                        "relevance"
                    ),
                )
            )

        self.item_popularity.cache().count()

    def refit(self, log: DataFrame, previous_log: Optional[Union[str, DataFrame]] = None, merged_log_path: Optional[str] = None) -> None:

        if self.use_relevance:

            self.item_abs_relevances = (
                log.select("item_idx", "relevance")
                .union(self.item_abs_relevances)
                .groupBy("item_idx")
                .agg(sf.sum("relevance").alias("relevance"))
            )

            new_user_ids = log.select("user_idx").join(self.all_user_ids, on=["user_idx"], how="leftanti").distinct()
            self.all_user_ids = self.all_user_ids.union(new_user_ids)
            self._users_count = self._users_count + new_user_ids.count()

            self.item_popularity = (
                self.item_abs_relevances
                .withColumn(
                    "relevance", sf.col("relevance") / sf.lit(self._users_count)
                )
            )
        else:
            new_item_idx = log.select("item_idx", "user_idx").join(self.item_users.select("item_idx"), on=["item_idx"], how="leftanti").distinct()
            # item_idx int, user_idx array<int>
            new_item_users = (
                new_item_idx.groupBy("item_idx")
                .agg(sf.collect_set('user_idx').alias('user_idx'))
            )

            existing_item_idx = log.select("item_idx", "user_idx").join(self.item_users.select("item_idx"), on=["item_idx"], how="inner")
            existing_item_groups = (
                existing_item_idx.groupBy("item_idx")
                .agg(sf.collect_set('user_idx').alias('new_user_idx'))
            )

            # item_idx int, user_idx array<int>
            self.item_users = (
                self.item_users.alias("a")
                .join(existing_item_groups.alias("b"), on=["item_idx"], how="left")
                .select("item_idx", sf.col("a.user_idx").alias("user_idx"), sf.col("b.new_user_idx").alias("new_user_idx"))
                .select("item_idx", "user_idx", sf.coalesce("new_user_idx", sf.array().cast("array<integer>")).alias("new_user_idx")) # converts nulls to empty arrays
                .select("item_idx", sf.array_union("user_idx", "new_user_idx").alias("user_idx"))
            )

            self.item_users = self.item_users.union(new_item_users)

            new_user_ids = log.select("user_idx").join(self.all_user_ids, on=["user_idx"], how="leftanti").distinct()
            self.all_user_ids.union(new_user_ids)
            self._users_count = self._users_count + new_user_ids.count()

            self.item_popularity = (
                self.item_users
                .select(
                    "item_idx",
                    (sf.size("user_idx") / sf.lit(self._users_count)).alias(
                        "relevance"
                    ),
                )
            )

        self.item_popularity = self.item_popularity.cache()
        self.item_popularity.write.mode("overwrite").format("noop").save()

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

        return self._predict_without_sampling(
            log, k, users, items, filter_seen_items
        )

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:

        return pairs.join(self.item_popularity, on="item_idx", how="inner")
