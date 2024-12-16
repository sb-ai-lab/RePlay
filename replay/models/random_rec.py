from typing import Optional

from replay.data import Dataset
from replay.utils import PYSPARK_AVAILABLE

from .base_rec import NonPersonalizedRecommender

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf


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

    >>> from replay.utils.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> state = State(spark)

    >>> import pandas as pd
    >>> from replay.data.dataset import Dataset, FeatureSchema, FeatureInfo, FeatureHint, FeatureType
    >>> from replay.utils.spark_utils import convert2spark
    >>>
    >>> interactions = convert2spark(pd.DataFrame({
    ...     "user_id": [1, 1, 2, 2, 3, 4],
    ...     "item_id": [1, 2, 2, 3, 3, 3]
    ... }))
    >>> interactions.show()
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
    ValueError: distribution can be one of [popular_based, relevance, uniform]

    >>> feature_schema = FeatureSchema(
    ...     [
    ...         FeatureInfo(
    ...             column="user_id",
    ...             feature_type=FeatureType.CATEGORICAL,
    ...             feature_hint=FeatureHint.QUERY_ID,
    ...         ),
    ...         FeatureInfo(
    ...             column="item_id",
    ...             feature_type=FeatureType.CATEGORICAL,
    ...             feature_hint=FeatureHint.ITEM_ID,
    ...         ),
    ...     ]
    ... )
    >>> dataset = Dataset(feature_schema, interactions)
    >>> random_pop = RandomRec(distribution="popular_based", alpha=1.0, seed=777)
    >>> random_pop.fit(dataset)
    >>> random_pop.item_popularity.show()
    +-------+------------------+
    |item_id|            rating|
    +-------+------------------+
    |      1|0.2222222222222222|
    |      2|0.3333333333333333|
    |      3|0.4444444444444444|
    +-------+------------------+
    <BLANKLINE>
    >>> recs = random_pop.predict(dataset, 2)
    >>> recs.show()
    +-------+-------+-------------------+
    |user_id|item_id|             rating|
    +-------+-------+-------------------+
    |      1|      3| 0.5403400662326382|
    |      2|      1| 0.8999281214873164|
    |      3|      1|  0.909252584485465|
    |      3|      2|0.14871615041196307|
    |      4|      2| 0.9923148665602072|
    |      4|      1| 0.4814574953690183|
    +-------+-------+-------------------+
    <BLANKLINE>
    >>> recs = random_pop.predict(dataset, 2, queries=[1], items=[7, 8])
    >>> recs.show()
    +-------+-------+-------------------+
    |user_id|item_id|             rating|
    +-------+-------+-------------------+
    |      1|      7|0.25547770543750425|
    |      1|      8|0.12755340075617772|
    +-------+-------+-------------------+
    <BLANKLINE>
    >>> random_pop = RandomRec(seed=555)
    >>> random_pop.fit(dataset)
    >>> random_pop.item_popularity.show()
    +-------+------------------+
    |item_id|            rating|
    +-------+------------------+
    |      1|0.3333333333333333|
    |      2|0.3333333333333333|
    |      3|0.3333333333333333|
    +-------+------------------+
    <BLANKLINE>
    """

    _search_space = {
        "distribution": {
            "type": "categorical",
            "args": ["popular_based", "relevance", "uniform"],
        },
        "alpha": {"type": "uniform", "args": [-0.5, 100]},
    }
    sample: bool = True

    def __init__(
        self,
        distribution: str = "uniform",
        alpha: float = 0.0,
        seed: Optional[int] = None,
        add_cold_items: bool = True,
        cold_weight: float = 0.5,
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
            msg = "distribution can be one of [popular_based, relevance, uniform]"
            raise ValueError(msg)
        if alpha <= -1.0 and distribution == "popular_based":
            msg = "alpha must be bigger than -1"
            raise ValueError(msg)
        self.distribution = distribution
        self.alpha = alpha
        self.seed = seed
        super().__init__(add_cold_items=add_cold_items, cold_weight=cold_weight)

    @property
    def _init_args(self):
        return {
            "distribution": self.distribution,
            "alpha": self.alpha,
            "seed": self.seed,
            "add_cold_items": self.add_cold_items,
            "cold_weight": self.cold_weight,
        }

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        if self.rating_column is None:
            self.rating_column = "rating"
        if self.distribution == "popular_based":
            self.item_popularity = (
                dataset.interactions.groupBy(self.item_column)
                .agg(sf.countDistinct(self.query_column).alias("user_count"))
                .select(
                    sf.col(self.item_column),
                    (sf.col("user_count").astype("float") + sf.lit(self.alpha)).alias(self.rating_column),
                )
            )
        elif self.distribution == "relevance":
            self.item_popularity = (
                dataset.interactions.groupBy(self.item_column)
                .agg(sf.sum(self.rating_column).alias(self.rating_column))
                .select(self.item_column, self.rating_column)
            )
        else:
            self.item_popularity = (
                dataset.interactions.select(self.item_column).distinct().withColumn(self.rating_column, sf.lit(1.0))
            )
        self.item_popularity = self.item_popularity.withColumn(
            self.rating_column,
            sf.col(self.rating_column) / self.item_popularity.agg(sf.sum(self.rating_column)).first()[0],
        )
        self.item_popularity.cache().count()
        self.fill = self._calc_fill(self.item_popularity, self.cold_weight, self.rating_column)
