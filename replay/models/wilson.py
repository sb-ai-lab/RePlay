from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from scipy.stats import norm

from replay.models.pop_rec import PopRec


class Wilson(PopRec):
    """
    Calculates lower confidence bound for the confidence interval
    of true fraction of positive ratings.

    ``relevance`` must be converted to binary 0-1 form.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 2], "item_idx": [1, 2], "relevance": [1, 1]})
    >>> from replay.utils.spark_utils import convert2spark
    >>> data_frame = convert2spark(data_frame)
    >>> model = Wilson()
    >>> model.fit_predict(data_frame,k=1).toPandas()
       user_idx  item_idx  relevance
    0         1         2   0.206549
    1         2         1   0.206549

    """

    # pylint: disable=too-many-arguments
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
            If true, cold items are assigned relevance equals to the less relevant item relevance
            multiplied by cold_weight and may appear among top-K recommendations.
            Otherwise cold items are filtered out.
            Could be changed after model training by setting the `add_cold_items` attribute.
        : param cold_weight: if `add_cold_items` is True,
            cold items are added with reduced relevance.
            The relevance for cold items is equal to the relevance
            of a least relevant item multiplied by a `cold_weight` value.
            `Cold_weight` value should be in interval (0, 1].
        :param sample: flag to choose recommendation strategy.
            If True, items are sampled with a probability proportional
            to the calculated predicted relevance.
            Could be changed after model training by setting the `sample` attribute.
        :param seed: random seed. Provides reproducibility if fixed
        """
        self.alpha = alpha
        self.sample = sample
        self.seed = seed
        super().__init__(
            add_cold_items=add_cold_items, cold_weight=cold_weight
        )

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
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:

        self._check_relevance(log)

        items_counts = log.groupby("item_idx").agg(
            sf.sum("relevance").alias("pos"),
            sf.count("relevance").alias("total"),
        )
        # https://en.wikipedia.org/w/index.php?title=Binomial_proportion_confidence_interval
        crit = norm.isf(self.alpha / 2.0)
        items_counts = items_counts.withColumn(
            "relevance",
            (sf.col("pos") + sf.lit(0.5 * crit**2))
            / (sf.col("total") + sf.lit(crit**2))
            - sf.lit(crit)
            / (sf.col("total") + sf.lit(crit**2))
            * sf.sqrt(
                (sf.col("total") - sf.col("pos"))
                * sf.col("pos")
                / sf.col("total")
                + crit**2 / 4
            ),
        )

        self.item_popularity = items_counts.drop("pos", "total")
        self.item_popularity.cache().count()
        self.fill = self._calc_fill(self.item_popularity, self.cold_weight)
