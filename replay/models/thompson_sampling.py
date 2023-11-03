from typing import Optional

import numpy as np
from pyspark.sql import functions as sf
from replay.data.dataset import Dataset

from replay.models.base_rec import NonPersonalizedRecommender


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
        super().__init__(add_cold_items=True, cold_weight=1)

    @property
    def _init_args(self):
        return {"sample": self.sample, "seed": self.seed}

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        self._check_rating(dataset)

        num_positive = dataset.interactions.filter(
            sf.col(self.rating_column) == sf.lit(1)
        ).groupby(self.item_column).agg(
            sf.count(self.rating_column).alias("positive")
        )
        num_negative = dataset.interactions.filter(
            sf.col(self.rating_column) == sf.lit(0)
        ).groupby(self.item_column).agg(
            sf.count(self.rating_column).alias("negative")
        )

        self.item_popularity = num_positive.join(
            num_negative, how="inner", on=self.item_column
        )

        self.item_popularity = self.item_popularity.withColumn(
            self.rating_column,
            sf.udf(np.random.beta, "double")("positive", "negative")
        ).drop("positive", "negative")
        self.item_popularity.cache().count()
        self.fill = np.random.beta(1, 1)
