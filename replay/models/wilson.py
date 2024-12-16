from typing import Optional

from scipy.stats import norm

from replay.data import Dataset
from replay.utils import PYSPARK_AVAILABLE

from .pop_rec import PopRec

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf


class Wilson(PopRec):
    """
    Calculates lower confidence bound for the confidence interval
    of true fraction of positive ratings.

    ``rating`` must be converted to binary 0-1 form.

    >>> import pandas as pd
    >>> from replay.data.dataset import Dataset, FeatureSchema, FeatureInfo, FeatureHint, FeatureType
    >>> from replay.utils.spark_utils import convert2spark
    >>> data_frame = pd.DataFrame({"user_id": [1, 2], "item_id": [1, 2], "rating": [1, 1]})
    >>> interactions = convert2spark(data_frame)
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
    ...         FeatureInfo(
    ...             column="rating",
    ...             feature_type=FeatureType.NUMERICAL,
    ...             feature_hint=FeatureHint.RATING,
    ...         ),
    ...     ]
    ... )
    >>> dataset = Dataset(feature_schema, interactions)
    >>> model = Wilson()
    >>> model.fit_predict(dataset, k=1).toPandas()
        user_id   item_id     rating
    0         1         2   0.206549
    1         2         1   0.206549

    """

    def __init__(
        self,
        alpha=0.05,
        add_cold_items: bool = True,
        cold_weight: float = 0.5,
        sample: bool = False,
        seed: Optional[int] = None,
    ):
        """
        :param alpha: significance level, default 0.05
        :param add_cold_items: flag to consider cold items in recommendations building
            if present in `items` parameter of `predict` method
            or `pairs` parameter of `predict_pairs` methods.
            If true, cold items are assigned rating equals to the less relevant item rating
            multiplied by cold_weight and may appear among top-K recommendations.
            Otherwise cold items are filtered out.
            Could be changed after model training by setting the `add_cold_items` attribute.
        : param cold_weight: if `add_cold_items` is True,
            cold items are added with reduced rating.
            The rating for cold items is equal to the rating
            of a least relevant item multiplied by a `cold_weight` value.
            `Cold_weight` value should be in interval (0, 1].
        :param sample: flag to choose recommendation strategy.
            If True, items are sampled with a probability proportional
            to the calculated predicted rating.
            Could be changed after model training by setting the `sample` attribute.
        :param seed: random seed. Provides reproducibility if fixed
        """
        self.alpha = alpha
        self.sample = sample
        self.seed = seed
        super().__init__(add_cold_items=add_cold_items, cold_weight=cold_weight)

    @property
    def _init_args(self):
        return {
            "alpha": self.alpha,
            "add_cold_items": self.add_cold_items,
            "cold_weight": self.cold_weight,
            "sample": self.sample,
            "seed": self.seed,
        }

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        self._check_rating(dataset)

        items_counts = dataset.interactions.groupby(self.item_column).agg(
            sf.sum(self.rating_column).alias("pos"),
            sf.count(self.rating_column).alias("total"),
        )
        # https://en.wikipedia.org/w/index.php?title=Binomial_proportion_confidence_interval
        crit = norm.isf(self.alpha / 2.0)
        items_counts = items_counts.withColumn(
            self.rating_column,
            (sf.col("pos") + sf.lit(0.5 * crit**2)) / (sf.col("total") + sf.lit(crit**2))
            - sf.lit(crit)
            / (sf.col("total") + sf.lit(crit**2))
            * sf.sqrt((sf.col("total") - sf.col("pos")) * sf.col("pos") / sf.col("total") + crit**2 / 4),
        )

        self.item_popularity = items_counts.drop("pos", "total")
        self.item_popularity.cache().count()
        self.fill = self._calc_fill(self.item_popularity, self.cold_weight, self.rating_column)
