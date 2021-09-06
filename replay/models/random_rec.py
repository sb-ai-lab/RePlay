from typing import Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import Recommender
from replay.constants import IDX_REC_SCHEMA


class RandomRec(Recommender):
    """
    Recommend random items, either weighted by item popularity or uniform.

    .. math::
        P\\left(i\\right)\\propto N_i + \\alpha

    :math:`N_i` --- number of users who rated item :math:`i`

    :math:`\\alpha` --- bigger :math:`\\alpha` values increase amount of rare items in recommendations.
        Must be bigger than -1. Default value is :math:`\\alpha = 0`.

    >>> from replay.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> state = State(spark)

    >>> import pandas as pd
    >>> from replay.utils import convert2spark
    >>>
    >>> log = convert2spark(pd.DataFrame({
    ...     "user_id": ["1", "1", "2", "2", "3", "4"],
    ...     "item_id": ["1", "2", "2", "3", "3", "3"]
    ... }))
    >>> log.show()
    +-------+-------+
    |user_id|item_id|
    +-------+-------+
    |      1|      1|
    |      1|      2|
    |      2|      2|
    |      2|      3|
    |      3|      3|
    |      4|      3|
    +-------+-------+
    <BLANKLINE>
    >>> random_pop = RandomRec(distribution="popular_based", alpha=-1)
    Traceback (most recent call last):
     ...
    ValueError: alpha must be bigger than -1

    >>> random_pop = RandomRec(distribution="abracadabra")
    Traceback (most recent call last):
     ...
    ValueError: distribution can be either popular_based or uniform

    >>> random_pop = RandomRec(distribution="popular_based", alpha=1.0, seed=777)
    >>> random_pop.fit(log)
    >>> random_pop.item_popularity.show()
    +--------+-----------+
    |item_idx|probability|
    +--------+-----------+
    |       2|        2.0|
    |       1|        3.0|
    |       0|        4.0|
    +--------+-----------+
    <BLANKLINE>
    >>> recs = random_pop.predict(log, 2)
    >>> recs.show()
    +-------+-------+------------------+
    |user_id|item_id|         relevance|
    +-------+-------+------------------+
    |      1|      3|0.3333333333333333|
    |      2|      1|               0.5|
    |      3|      1|               1.0|
    |      3|      2|0.3333333333333333|
    |      4|      2|               0.5|
    |      4|      1|0.3333333333333333|
    +-------+-------+------------------+
    <BLANKLINE>
    >>> recs = random_pop.predict(log, 2, users=[1], items=[7, 8])
    >>> recs.show()
    +-------+-------+---------+
    |user_id|item_id|relevance|
    +-------+-------+---------+
    |      1|      7|      1.0|
    |      1|      8|      0.5|
    +-------+-------+---------+
    <BLANKLINE>
    >>> random_pop = RandomRec(seed=555)
    >>> random_pop.fit(log)
    >>> random_pop.item_popularity.show()
    +--------+-----------+
    |item_idx|probability|
    +--------+-----------+
    |       2|          1|
    |       1|          1|
    |       0|          1|
    +--------+-----------+
    <BLANKLINE>
    >>> recs = random_pop.predict(log, 2)
    >>> recs.show()
    +-------+-------+------------------+
    |user_id|item_id|         relevance|
    +-------+-------+------------------+
    |      1|      3|               1.0|
    |      2|      1|               0.5|
    |      3|      2|               0.5|
    |      3|      1|0.3333333333333333|
    |      4|      1|               1.0|
    |      4|      2|               0.5|
    +-------+-------+------------------+
    <BLANKLINE>
    """

    item_popularity: DataFrame
    fill: float
    can_predict_cold_users = True
    can_predict_cold_items = True
    _search_space = {
        "distribution": {
            "type": "categorical",
            "args": ["popular_based", "uniform"],
        },
        "alpha": {"type": "uniform", "args": [-0.5, 100]},
    }

    def __init__(
        self,
        distribution: str = "uniform",
        alpha: float = 0.0,
        seed: Optional[int] = None,
        add_cold: Optional[bool] = True,
    ):
        """
        :param distribution: recommendation strategy:
            "uniform" - all items are sampled uniformly
            "popular_based" - recommend popular items more
        :param alpha: bigger values adjust model towards less popular items
        :param seed: random seed
        :param add_cold: flag to add cold items with minimal probability
        """
        if distribution not in ("popular_based", "uniform"):
            raise ValueError(
                "distribution can be either popular_based or uniform"
            )
        if alpha <= -1.0 and distribution == "popular_based":
            raise ValueError("alpha must be bigger than -1")
        self.distribution = distribution
        self.alpha = alpha
        self.seed = seed
        self.add_cold = add_cold

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        if self.distribution == "popular_based":
            probability = f"CAST(user_count + {self.alpha} AS DOUBLE)"
        else:
            probability = "1"
        self.item_popularity = log.groupBy("item_idx").agg(
            sf.countDistinct("user_idx").alias("user_count")
        )
        self.item_popularity = self.item_popularity.selectExpr(
            "item_idx", f"{probability} AS probability"
        )
        self.item_popularity.cache()
        if self.add_cold:
            fill = self.item_popularity.agg({"probability": "min"}).first()[0]
        else:
            fill = 0
        self.fill = fill

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
                how="left",
            )
            .drop("item")
            .fillna(self.fill)
            .toPandas()
        )
        items_pd.loc[:, "probability"] = (
            items_pd["probability"] / items_pd["probability"].sum()
        )
        seed = self.seed

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            user_idx = pandas_df["user_idx"][0]
            cnt = pandas_df["cnt"][0]
            if seed is not None:
                np.random.seed(user_idx + seed)
            items_idx = np.random.choice(
                items_pd["item_idx"].values,
                size=cnt,
                p=items_pd["probability"].values,
                replace=False,
            )
            relevance = 1 / np.arange(1, cnt + 1)
            return pd.DataFrame(
                {
                    "user_idx": cnt * [user_idx],
                    "item_idx": items_idx,
                    "relevance": relevance,
                }
            )

        model_len = len(items_pd)
        recs = (
            users.join(log, how="left", on="user_idx")
            .select("user_idx", "item_idx")
            .groupby("user_idx")
            .agg(sf.countDistinct("item_idx").alias("cnt"))
            .selectExpr(
                "user_idx",
                f"LEAST(cnt + {k}, {model_len}) AS cnt",
            )
            .groupby("user_idx")
            .applyInPandas(grouped_map, IDX_REC_SCHEMA)
        )

        return recs
