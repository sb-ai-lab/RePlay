from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import NonPersonalizedRecommender
from replay.utils import unpersist_after, unionify


# pylint: disable=too-many-instance-attributes
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
    +--------+------------------+
    |item_idx|         relevance|
    +--------+------------------+
    |       1|0.2222222222222222|
    |       2|0.3333333333333333|
    |       3|0.4444444444444444|
    +--------+------------------+
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

    _search_space = {
        "distribution": {
            "type": "categorical",
            "args": ["popular_based", "relevance", "uniform"],
        },
        "alpha": {"type": "uniform", "args": [-0.5, 100]},
    }
    fill: float

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        distribution: str = "uniform",
        alpha: float = 0.0,
        seed: Optional[int] = None,
        add_cold_items: bool = True,
        cold_weight: float = 0.5,
        sample: bool = False,
    ):
        """
        :param distribution: recommendation strategy:
            "uniform" - all items are sampled uniformly
            "popular_based" - recommend popular items more
        :param alpha: bigger values adjust model towards less popular items
        :param seed: random seed
        :param add_cold_items: flag to consider cold items in recommendations building
            if present in `items` parameter of `predict` method
            or `pairs` parameter of `predict_pairs` methods.
            If true, cold items are assigned relevance equals to the less relevant item relevance
            multiplied by `cold_weight` and may appear among top-K recommendations.
            Otherwise cold items are filtered out.
            Could be changed after model training by setting the `add_cold_items` attribute.
        :param cold_weight: if `add_cold_items` is True,
            cold items are added with reduced relevance.
            The relevance for cold items is equal to the relevance
            of a least relevant item multiplied by a `cold_weight` value.
            `Cold_weight` value should be in interval (0, 1].
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
        self.total_relevance = 0.0
        self.relevance_sums: Optional[DataFrame] = None
        self.item_popularity: Optional[DataFrame] = None
        self.sample = sample
        super().__init__(
            add_cold_items=add_cold_items, cold_weight=cold_weight
        )

    @property
    def _init_args(self):
        return {
            "distribution": self.distribution,
            "alpha": self.alpha,
            "seed": self.seed,
            "add_cold_items": self.add_cold_items,
            "cold_weight": self.cold_weight,
        }

    @property
    def _dataframes(self):
        return {
            "item_popularity": self.item_popularity,
            "relevance_sums": self.relevance_sums,
        }

    def _clear_cache(self):
        for df in self._dataframes.values():
            if df is not None:
                df.unpersist()

    def _load_model(self, path: str):
        if self.add_cold_items:
            fill = self.item_popularity.agg({"relevance": "min"}).first()[0]
        else:
            fill = 0
        self.fill = fill

    def _fit_partial(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        previous_log: Optional[DataFrame] = None,
    ) -> None:
        with unpersist_after(self._dataframes):
            if self.distribution == "popular_based":
                # storing the intermediate aggregate (e.g. agg result) is
                # almost as costly as storing the whole previous
                # due to amount of unique pairs in previous_log should approximately equal
                # to the number of entries in previous_log at all
                item_popularity = (
                    unionify(log, previous_log)
                    .groupBy("item_idx")
                    .agg(sf.countDistinct("user_idx").alias("user_count"))
                    .select(
                        "item_idx",
                        (
                            sf.col("user_count").astype("float")
                            + sf.lit(self.alpha)
                        ).alias("relevance"),
                    )
                )
            elif self.distribution == "relevance":
                self.relevance_sums = (
                    unionify(
                        log.select("item_idx", "relevance"),
                        self.relevance_sums,
                    )
                    .groupBy("item_idx")
                    .agg(sf.sum("relevance").alias("relevance"))
                    .cache()
                )
                item_popularity = self.relevance_sums
            else:
                item_popularity = unionify(
                    log.select("item_idx", sf.lit(1.0).alias("relevance")),  # pylint: disable=no-member
                    self.item_popularity,
                ).drop_duplicates(["item_idx"])

            self.item_popularity = item_popularity.select(
                "item_idx",
                (
                    sf.col("relevance")
                    / item_popularity.agg(sf.sum("relevance")).first()[0]
                ).alias("relevance"),
            )

            self.item_popularity.cache().count()
            self.fill = self._calc_fill(self.item_popularity, self.cold_weight)
