from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import NonPersonalizedRecommender
from replay.utils import unionify, unpersist_after


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

    sample: bool = False

    def __init__(
        self,
        use_relevance: bool = False,
        add_cold_items: bool = True,
        cold_weight: float = 0.5,
    ):
        """
        :param use_relevance: flag to use relevance values as is or to treat them as 1
        :param add_cold_items: flag to consider cold items in recommendations building
            if present in `items` parameter of `predict` method
            or `pairs` parameter of `predict_pairs` methods.
            If true, cold items are assigned relevance equals to the less relevant item relevance
            multiplied by cold_weight and may appear among top-K recommendations.
            Otherwise cold items are filtered out.
            Could be changed after model training by setting the `add_cold_items` attribute.
        : param cold_weight: if `add_cold_items` is True,
            cold items are added with reduced relevance.
            The relevance for cold items is equal to the relevance
            of a least relevant item multiplied by a `cold_weight` value.
            `Cold_weight` value should be in interval (0, 1].
        """
        self.use_relevance = use_relevance
        self.all_user_ids: Optional[DataFrame] = None
        self.item_abs_relevances: Optional[DataFrame] = None
        self.item_popularity: Optional[DataFrame] = None
        super().__init__(
                add_cold_items=add_cold_items, cold_weight=cold_weight
            )

    def copy(self):
        return PopRec(
            use_relevance=self.use_relevance,
            add_cold_items=self.add_cold_items,
            cold_weight=self.cold_weight
        )

    @property
    def _init_args(self):
        return {
            "use_relevance": self.use_relevance,
            "add_cold_items": self.add_cold_items,
            "cold_weight": self.cold_weight,
        }
    
    @property
    def _dataframes(self):
        return {
            "all_user_ids": self.all_user_ids,
            "item_abs_relevances": self.item_abs_relevances,
            "item_popularity": self.item_popularity
        }

    def _clear_cache(self):
        for df in self._dataframes.values():
            if df is not None:
                df.unpersist()

    def _fit_partial(
            self,
            log: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            previous_log: Optional[DataFrame] = None):
        with unpersist_after(self._dataframes):
            self.all_user_ids = unionify(log.select("user_idx"), self.all_user_ids).distinct().cache()
            self._users_count = self.all_user_ids.count()
            if self.use_relevance:
                # we will save it to update fitted model
                self.item_abs_relevances = (
                    unionify(log.select("item_idx", "relevance"), self.item_abs_relevances)
                    .groupBy("item_idx")
                    .agg(sf.sum("relevance").alias("relevance"))
                ).cache()

                self.item_popularity = (
                    self.item_abs_relevances.withColumn("relevance", sf.col("relevance") / sf.lit(self._users_count))
                )
            else:
                log = unionify(log, previous_log)
                # equal to storing a whole old log which may be huge
                self.item_popularity = (
                    log
                    .groupBy("item_idx")
                    .agg(
                        (sf.countDistinct("user_idx").alias("relevance") / sf.lit(self._users_count)).alias("relevance")
                    )
                )

            self.item_popularity.cache().count()
            self.fill = self._calc_fill(self.item_popularity, self.cold_weight)
