from typing import Optional, Union

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
    >>> from replay.utils import convert2spark
    >>> data_frame = convert2spark(data_frame)
    >>> model = Wilson()
    >>> model.fit_predict(data_frame,k=1).toPandas()
       user_idx  item_idx  relevance
    0         1         2   0.206549
    1         2         1   0.206549

    """

    def __init__(self, alpha=0.05):
        """
        :param alpha: significance level, default 0.05
        """
        # pylint: disable=super-init-not-called
        self.alpha = alpha

    @property
    def _init_args(self):
        return {"alpha": self.alpha}

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:

        self._check_relevance(log)

        self.items_counts_aggr = log.groupby("item_idx").agg(
            sf.sum("relevance").alias("pos"),
            sf.count("relevance").alias("total"),
        )
        # https://en.wikipedia.org/w/index.php?title=Binomial_proportion_confidence_interval
        crit = norm.isf(self.alpha / 2.0)
        items_counts = self.items_counts_aggr.withColumn(
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

    def refit(
        self,
        log: DataFrame,
        previous_log: Optional[Union[str, DataFrame]] = None,
        merged_log_path: Optional[str] = None,
    ) -> None:

        items_counts = log.groupby("item_idx").agg(
            sf.sum("relevance").alias("pos"),
            sf.count("relevance").alias("total"),
        )

        self.items_counts_aggr = (
            self.items_counts_aggr.union(items_counts)
            .groupby("item_idx")
            .agg(
                sf.sum("pos").alias("pos"),
                sf.sum("total").alias("total"),
            )
        )

        # https://en.wikipedia.org/w/index.php?title=Binomial_proportion_confidence_interval
        crit = norm.isf(self.alpha / 2.0)
        self.item_popularity = self.items_counts_aggr.select(
            "item_idx",
            (
                (sf.col("pos") + sf.lit(0.5 * crit**2))
                / (sf.col("total") + sf.lit(crit**2))
                - sf.lit(crit)
                / (sf.col("total") + sf.lit(crit**2))
                * sf.sqrt(
                    (sf.col("total") - sf.col("pos"))
                    * sf.col("pos")
                    / sf.col("total")
                    + crit**2 / 4
                )
            ).alias("relevance")
        )
        