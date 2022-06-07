import math

from typing import Optional

import pandas as pd

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf

from replay.models.base_rec import Recommender


class UCB(Recommender):
    """
    Calculates upper confidence bound (UCB) for the confidence interval
    of true fraction of positive ratings.

    ``relevance`` from log must be converted to binary 0-1 form.

    .. math::
        pred_i = \\overline rating \\sqrt{\\frac{\\ln{c\\cdot n}}{n_i}}

    :math:`pred_i` -- predictive relevance items :math:`i`

    :math:`\\overline raiting` -- avarage rating of items :math:`i`

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 2, 3, 3], "item_idx": [1, 2, 1, 2], "relevance": [1, 0, 0, 0]})
    >>> from replay.utils import convert2spark
    >>> data_frame = convert2spark(data_frame)
    >>> model = UCB()
    >>> model.fit(data_frame)
    >>> model.predict(data_frame,k=2,users=[1,2,3,4], items=[1,2,3]).toPandas()
       user_idx  item_idx  relevance
    0         3         3   2.442027
    1         1         3   2.442027
    2         1         2   1.019667
    3         2         3   2.442027
    4         2         1   1.519667
    5         4         3   2.442027
    6         4         2   1.519667

    """
    can_predict_cold_users = True
    can_predict_cold_items = True
    fill: float

    def __init__(self, c=2):
        """
        :param c: exploration coefficient
        """
        # pylint: disable=super-init-not-called
        self.c = c

    def _fit(
            self,
            log: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
    ) -> None:
        vals = log.select("relevance").where(
            (sf.col("relevance") != 1) & (sf.col("relevance") != 0)
        )
        if vals.count() > 0:
            raise ValueError("Relevance values in log must be 0 or 1")

        items_counts = log.groupby("item_idx").agg(
            sf.sum("relevance").alias("pos"),
            sf.count("relevance").alias("total"),
        )

        full_count = log.count()
        items_counts = items_counts.withColumn(
            "relevance",
            (sf.col("pos") / sf.col("total") + sf.sqrt(sf.log(sf.lit(self.c *
                                                                     full_count)) / sf.col("total")))
        )

        self.item_popularity = items_counts.drop("pos", "total")
        self.item_popularity.cache()

        self.fill = 1 + math.sqrt(math.log(self.c * full_count))

    @property
    def _dataframes(self):
        return {"item_popularity": self.item_popularity}

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
            how="right",
        ).fillna(value=self.fill, subset=["relevance"]).withColumn(
            "rank",
            sf.row_number().over(Window.orderBy(sf.col("relevance").desc(),
                                                sf.col("item_idx").desc())),
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
        return pairs.join(self.item_popularity, on="item_idx", how="right")\
            .fillna(value=self.fill, subset=["relevance"])
