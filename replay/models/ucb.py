from typing import Optional

import pandas as pd

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf

from replay.models.wilson import Wilson


class UCB(Wilson):
    """
    Calculates upper confidence bound for the confidence interval
    of true fraction of positive ratings.

    ``relevance`` must be converted to binary 0-1 form.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 2], "item_idx": [1, 2], "relevance": [1, 1]})
    >>> from replay.utils import convert2spark
    >>> data_frame = convert2spark(data_frame)
    >>> model = UCB()
    >>> model.fit_predict(data_frame,k=1).toPandas()
       user_idx  item_idx  relevance
    0         1         2    2.17741
    1         2         1    2.17741

    """

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
            (sf.col("pos") / sf.col("total") + sf.sqrt(sf.log(sf.lit(2 * full_count)) / sf.col("total")))
        )

        self.item_popularity = items_counts.drop("pos", "total")
        self.item_popularity.cache()

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
        ).fillna(value=1, subset=["relevance"]).withColumn(
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
            .fillna(value=1, subset=["relevance"])
