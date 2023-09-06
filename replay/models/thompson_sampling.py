from typing import Optional

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import NonPersonalizedRecommender
from replay.utils.spark_utils import unpersist_after, unionify


class ThompsonSampling(NonPersonalizedRecommender):
    """
    Thompson Sampling recommender.

    Bandit model with `efficient exploration-exploitation balance
    <https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf>`_.
    The reward probability of each of the K arms is modeled by a Beta distribution
    which is updated after an arm is selected. The initial prior distribution is Beta(1,1).
    """

    def __init__(
        self,
        sample: bool = False,
        seed: Optional[int] = None,
    ):
        self.sample = sample
        self.seed = seed
        self.num_positive: Optional[DataFrame] = None
        self.num_negative: Optional[DataFrame] = None
        self.item_popularity: Optional[DataFrame] = None
        super().__init__(add_cold_items=True, cold_weight=1)

    @property
    def _init_args(self):
        return {"sample": self.sample, "seed": self.seed}

    # def _fit(
    #     self,
    #     log: DataFrame,
    #     user_features: Optional[DataFrame] = None,
    #     item_features: Optional[DataFrame] = None,
    # ) -> None:
    #     self._check_relevance(log)
    #
    #     num_positive = log.filter(
    #         log.relevance == sf.lit(1)
    #     ).groupby("item_idx").agg(
    #         sf.count("relevance").alias("positive")
    #     )
    #     num_negative = log.filter(
    #         log.relevance == sf.lit(0)
    #     ).groupby("item_idx").agg(
    #         sf.count("relevance").alias("negative")
    #     )
    #
    #     self.item_popularity = num_positive.join(
    #         num_negative, how="inner", on="item_idx"
    #     )
    #
    #     self.item_popularity = self.item_popularity.withColumn(
    #         "relevance",
    #         sf.udf(np.random.beta, "double")("positive", "negative")
    #     ).drop("positive", "negative")
    #     self.item_popularity.cache().count()
    #     self.fill = np.random.beta(1, 1)

    def _fit_partial(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        previous_log: Optional[DataFrame] = None,
    ) -> None:
        with unpersist_after(self._dataframes):
            self._check_relevance(log)
            if previous_log:
                self._check_relevance(previous_log)

            # we save this dataframes for the refit() method
            self.num_positive = (
                unionify(
                    log.filter(log.relevance == sf.lit(1)).select(
                        "item_idx",
                        sf.lit(1).alias(  # pylint: disable=no-member
                            "positive"
                        ),
                    ),
                    self.num_positive,
                )
                .groupby("item_idx")
                .agg(sf.sum("positive").alias("positive"))
                .cache()
            )
            self.num_negative = (
                unionify(
                    log.filter(log.relevance == sf.lit(0)).select(
                        "item_idx",
                        sf.lit(1).alias(  # pylint: disable=no-member
                            "negative"
                        ),
                    ),
                    self.num_negative,
                )
                .groupby("item_idx")
                .agg(sf.sum("negative").alias("negative"))
                .cache()
            )

            self.item_popularity = self.num_positive.join(
                self.num_negative, how="inner", on="item_idx"
            )

            self.item_popularity = self.item_popularity.withColumn(
                "relevance",
                sf.udf(np.random.beta, "double")("positive", "negative"),
            ).drop("positive", "negative")
            self.item_popularity.cache().count()
            self.fill = np.random.beta(1, 1)
