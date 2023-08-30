import math

from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.metrics import Metric, NDCG
from replay.models.base_rec import NonPersonalizedRecommender


class UCB(NonPersonalizedRecommender):
    """Simple bandit model, which caclulate item relevance as upper confidence bound
    (`UCB <https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-upper-confidence-bound-algorithm-4b84be516047>`_)
    for the confidence interval of true fraction of positive ratings.
    Should be used in iterative (online) mode to achive proper recommendation quality.

    ``relevance`` from log must be converted to binary 0-1 form.

    .. math::
        pred_i = ctr_i + \\sqrt{\\frac{c\\ln{n}}{n_i}}

    :math:`pred_i` -- predicted relevance of item :math:`i`
    :math:`c` -- exploration coeficient
    :math:`n` -- number of interactions in log
    :math:`n_i` -- number of interactions with item :math:`i`

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 2, 3, 3], "item_idx": [1, 2, 1, 2], "relevance": [1, 0, 0, 0]})
    >>> from replay.utils.spark_utils import convert2spark
    >>> data_frame = convert2spark(data_frame)
    >>> model = UCB()
    >>> model.fit(data_frame)
    >>> model.predict(data_frame,k=2,users=[1,2,3,4], items=[1,2,3]
    ... ).toPandas().sort_values(["user_idx","relevance","item_idx"],
    ... ascending=[True,False,True]).reset_index(drop=True)
       user_idx  item_idx  relevance
    0         1         3   2.665109
    1         1         2   1.177410
    2         2         3   2.665109
    3         2         1   1.677410
    4         3         3   2.665109
    5         4         3   2.665109
    6         4         1   1.677410

    """

    # attributes which are needed for refit method
    full_count: int
    items_counts_aggr: DataFrame

    def __init__(
        self,
        exploration_coef: float = 2,
        sample: bool = False,
        seed: Optional[int] = None,
    ):
        """
        :param exploration_coef: exploration coefficient
        :param sample: flag to choose recommendation strategy.
            If True, items are sampled with a probability proportional
            to the calculated predicted relevance.
            Could be changed after model training by setting the `sample` attribute.
        :param seed: random seed. Provides reproducibility if fixed
        """
        # pylint: disable=super-init-not-called
        self.coef = exploration_coef
        self.sample = sample
        self.seed = seed
        super().__init__(add_cold_items=True, cold_weight=1)

    @property
    def _init_args(self):
        return {
            "exploration_coef": self.coef,
            "sample": self.sample,
            "seed": self.seed,
        }

    # pylint: disable=too-many-arguments
    def optimize(
        self,
        train: DataFrame,
        test: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        param_borders: Optional[Dict[str, List[Any]]] = None,
        criterion: Metric = NDCG(),
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ) -> None:
        """
        Searches best parameters with optuna.

        :param train: train data
        :param test: test data
        :param user_features: user features
        :param item_features: item features
        :param param_borders: a dictionary with search borders, where
            key is the parameter name and value is the range of possible values
            ``{param: [low, high]}``. In case of categorical parameters it is
            all possible values: ``{cat_param: [cat_1, cat_2, cat_3]}``.
        :param criterion: metric to use for optimization
        :param k: recommendation list length
        :param budget: number of points to try
        :param new_study: keep searching with previous study or start a new study
        :return: dictionary with best parameters
        """
        self.logger.warning(
            "The UCB model has only exploration coefficient parameter, "
            "which cannot not be directly optimized"
        )

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:

        self._check_relevance(log)

        # we save this dataframe for the refit() method
        self.items_counts_aggr = log.groupby("item_idx").agg(
            sf.sum("relevance").alias("pos"),
            sf.count("relevance").alias("total"),
        )
        # we save this variable for the refit() method
        self.full_count = log.count()

        self._calc_item_popularity()

    def refit(
        self,
        log: DataFrame,
    ) -> None:
        """Iteratively refit with new part of log.

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :return:
        """

        self._check_relevance(log)

        # aggregate new log part
        items_counts_aggr = log.groupby("item_idx").agg(
            sf.sum("relevance").alias("pos"),
            sf.count("relevance").alias("total"),
        )
        # combine old and new aggregations and aggregate
        self.items_counts_aggr = (
            self.items_counts_aggr.union(items_counts_aggr)
            .groupby("item_idx")
            .agg(
                sf.sum("pos").alias("pos"),
                sf.sum("total").alias("total"),
            )
        )
        # sum old and new log lengths
        self.full_count += log.count()

        self._calc_item_popularity()

    def _calc_item_popularity(self):

        items_counts = self.items_counts_aggr.withColumn(
            "relevance",
            (
                sf.col("pos") / sf.col("total")
                + sf.sqrt(
                    self.coef
                    * sf.log(sf.lit(self.full_count))
                    / sf.col("total")
                )
            ),
        )

        self.item_popularity = items_counts.drop("pos", "total")
        self.item_popularity.cache().count()

        self.fill = 1 + math.sqrt(self.coef * math.log(self.full_count))
