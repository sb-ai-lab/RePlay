from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import NonPersonalizedRecommender


class RandomRec(NonPersonalizedRecommender):
    """
    Recommend random items, either weighted by item popularity or uniform.

    .. math::
        P\\left(i\\right)\\propto N_i + \\alpha

    :math:`N_i` --- number of users who rated item :math:`i`

    :math:`\\alpha` --- bigger :math:`\\alpha` values increase amount of rare items in recommendations.
        Must be bigger than -1. Default value is :math:`\\alpha = 0`.

    Model without seed provides non-determenistic recommendations,
    model with fixed seed provides reproducible recommendataions.

    As the recommendations from `predict` are cached, save them to disk, or create a checkpoint
    and unpersist them to get different recommendations after another `predict` call.

    >>> from replay.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> state = State(spark)

    >>> import pandas as pd
    >>> from replay.utils import convert2spark
    >>>
    >>> log = convert2spark(pd.DataFrame({
    ...     "user_idx": [1, 1, 2, 2, 3, 4],
    ...     "item_idx": [1, 2, 2, 3, 3, 3]
    ... }))
    >>> log.show()
    +--------+--------+
    |user_idx|item_idx|
    +--------+--------+
    |       1|       1|
    |       1|       2|
    |       2|       2|
    |       2|       3|
    |       3|       3|
    |       4|       3|
    +--------+--------+
    <BLANKLINE>
    >>> random_pop = RandomRec(distribution="popular_based", alpha=-1)
    Traceback (most recent call last):
     ...
    ValueError: alpha must be bigger than -1

    >>> random_pop = RandomRec(distribution="abracadabra")
    Traceback (most recent call last):
     ...
    ValueError: distribution can be one of [popular_based, relevance, uniform]

    >>> random_pop = RandomRec(distribution="popular_based", alpha=1.0, seed=777)
    >>> random_pop.fit(log)
    >>> random_pop.item_popularity.show()
    +--------+---------+
    |item_idx|relevance|
    +--------+---------+
    |       1|      2.0|
    |       2|      3.0|
    |       3|      4.0|
    +--------+---------+
    <BLANKLINE>
    >>> recs = random_pop.predict(log, 2)
    >>> recs.show()
    +--------+--------+------------------+
    |user_idx|item_idx|         relevance|
    +--------+--------+------------------+
    |       1|       3|0.3333333333333333|
    |       2|       1|               0.5|
    |       3|       2|               1.0|
    |       3|       1|0.3333333333333333|
    |       4|       2|               1.0|
    |       4|       1|               0.5|
    +--------+--------+------------------+
    <BLANKLINE>
    >>> recs = random_pop.predict(log, 2, users=[1], items=[7, 8])
    >>> recs.show()
    +--------+--------+---------+
    |user_idx|item_idx|relevance|
    +--------+--------+---------+
    |       1|       7|      1.0|
    |       1|       8|      0.5|
    +--------+--------+---------+
    <BLANKLINE>
    >>> random_pop = RandomRec(seed=555)
    >>> random_pop.fit(log)
    >>> random_pop.item_popularity.show()
    +--------+---------+
    |item_idx|relevance|
    +--------+---------+
    |       1|      1.0|
    |       2|      1.0|
    |       3|      1.0|
    +--------+---------+
    <BLANKLINE>
    """

    can_predict_cold_items = True
    _search_space = {
        "distribution": {
            "type": "categorical",
            "args": ["popular_based", "relevance", "uniform"],
        },
        "alpha": {"type": "uniform", "args": [-0.5, 100]},
    }
    fill: float

    def __init__(
        self,
        distribution: str = "uniform",
        alpha: float = 0.0,
        seed: Optional[int] = None,
        add_cold_items: Optional[bool] = True,
    ):
        """
        :param distribution: recommendation strategy:
            "uniform" - all items are sampled uniformly
            "popular_based" - recommend popular items more
        :param alpha: bigger values adjust model towards less popular items
        :param seed: random seed
        :param add_cold_items: flag to add cold items with minimal probability
        """
        if distribution not in ("popular_based", "relevance", "uniform"):
            raise ValueError(
                "distribution can be one of [popular_based, relevance, uniform]"
            )
        if alpha <= -1.0 and distribution == "popular_based":
            raise ValueError("alpha must be bigger than -1")
        self.distribution = distribution
        self.alpha = alpha
        self.seed = seed
        self.add_cold_items = add_cold_items

    @property
    def _init_args(self):
        return {
            "distribution": self.distribution,
            "alpha": self.alpha,
            "seed": self.seed,
            "add_cold_items": self.add_cold_items,
        }

    def _load_model(self, path: str):
        if self.add_cold_items:
            fill = self.item_popularity.agg({"relevance": "min"}).first()[0]
        else:
            fill = 0
        self.fill = fill

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:

        if self.distribution == "popular_based":
            self.item_popularity = (
                log.groupBy("item_idx")
                .agg(sf.countDistinct("user_idx").alias("user_count"))
                .select(
                    sf.col("item_idx"),
                    (sf.col("user_count").astype("float") + self.alpha).alias(
                        "relevance"
                    ),
                )
            )
        elif self.distribution == "relevance":
            total_relevance = log.agg(sf.sum("relevance")).first()[0]
            self.item_popularity = (
                log.groupBy("item_idx")
                .agg(sf.sum("relevance").alias("relevance"))
                .select(
                    "item_idx",
                    (sf.col("relevance") / sf.lit(total_relevance)).alias(
                        "relevance"
                    ),
                )
            )
        else:
            self.item_popularity = (
                log.select("item_idx")
                .distinct()
                .withColumn("relevance", sf.lit(1.0))
            )

        self.item_popularity.cache().count()
        self.fill = (
            self.item_popularity.agg({"relevance": "min"}).first()[0]
            if self.add_cold_items
            else 0.0
        )

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

        return self._predict_with_sampling(
            log, k, users, items, filter_seen_items, self.add_cold_items
        )
