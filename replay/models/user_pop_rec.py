from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import Recommender, PartialFitMixin
from replay.utils.spark_utils import unionify, unpersist_after


class UserPopRec(Recommender, PartialFitMixin):
    """
    Recommends old objects from each user's personal top.
    Input is the number of interactions between users and items.

    Popularity for item :math:`i` and user :math:`u` is defined as the
    fraction of actions with item :math:`i` among all interactions of user :math:`u`:

    .. math::
        Popularity(i_u) = \\dfrac{N_iu}{N_u}

    :math:`N_iu` - number of interactions of user :math:`u` with item :math:`i`.
    :math:`N_u` - total number of interactions of user :math:`u`.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 1, 3], "item_idx": [1, 2, 3], "relevance": [2, 1, 1]})
    >>> data_frame
       user_idx  item_idx  relevance
    0         1         1          2
    1         1         2          1
    2         3         3          1

    >>> from replay.utils.spark_utils import convert2spark
    >>> data_frame = convert2spark(data_frame)
    >>> model = UserPopRec()
    >>> res = model.fit_predict(data_frame, 1, filter_seen_items=False)
    >>> model.user_item_popularity.count()
    3
    >>> res.toPandas().sort_values("user_idx", ignore_index=True)
       user_idx  item_idx  relevance
    0         1         1   0.666667
    1         3         3   1.000000
    """

    def __init__(self):
        super().__init__()
        self.user_relevance_sum: Optional[DataFrame] = None
        self.user_item_popularity: Optional[DataFrame] = None

    @property
    def _init_args(self):
        return {}

    @property
    def _dataframes(self):
        return {
            "user_item_popularity": self.user_item_popularity,
            "user_relevance_sum": self.user_relevance_sum,
        }

    def _fit_partial(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        previous_log: Optional[DataFrame] = None,
    ) -> None:
        with unpersist_after(self._dataframes):
            # new users: just unionify(user_item_popularity, self.user_item_popularity)
            # new items: groupby users +

            relevance_sums = log.groupBy("user_idx", "item_idx").agg(
                sf.sum("relevance").alias("user_item_rel_sum")
            )

            self.user_relevance_sum = (
                unionify(
                    relevance_sums.select(
                        "user_idx",
                        sf.col("user_item_rel_sum").alias("user_rel_sum"),
                    ),
                    self.user_relevance_sum,
                )
                .groupBy("user_idx")
                .agg(sf.sum("user_rel_sum").alias("user_rel_sum"))
            ).cache()

            self.user_item_popularity = relevance_sums.join(
                self.user_relevance_sum,
                how="inner",
                on="user_idx",
            ).select(
                "user_idx",
                "item_idx",
                (sf.col("user_item_rel_sum") / sf.col("user_rel_sum")).alias(
                    "relevance"
                ),
            )
            self.user_item_popularity.cache().count()

    def _clear_cache(self):
        for df in self._dataframes.values():
            if df is not None:
                df.unpersist()

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
        if filter_seen_items:
            self.logger.warning(
                "UserPopRec can't predict new items, recommendations will not be filtered"
            )

        return self.user_item_popularity.join(users, on="user_idx").join(
            items, on="item_idx"
        )
