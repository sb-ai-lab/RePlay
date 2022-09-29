import joblib
import math

from os.path import join
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.random import default_rng

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf

from replay.constants import REC_SCHEMA
from replay.metrics import Metric, NDCG
from replay.models.base_rec import Recommender


class UCB(Recommender):
    """Simple bandit model, which caclulate item relevance as upper confidence bound
    (`UCB <https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-upper-confidence-bound-algorithm-4b84be516047>`_)
    for the confidence interval of true fraction of positive ratings.
    Should be used in iterative (online) mode to achive preper recommendation quality.

    ``relevance`` from log must be converted to binary 0-1 form.

    .. math::
        pred_i = ctr_i + \\sqrt{\\frac{c\\ln{n}}{n_i}}

    :math:`pred_i` -- predicted relevance of item :math:`i`
    :math:`c` -- exploration coeficient
    :math:`n` -- number of interactions in log
    :math:`n_i` -- number of interactions with item :math:`i`

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 2, 3, 3], "item_idx": [1, 2, 1, 2], "relevance": [1, 0, 0, 0]})
    >>> from replay.utils import convert2spark
    >>> data_frame = convert2spark(data_frame)
    >>> model = UCB()
    >>> model.fit(data_frame)
    >>> model.predict(data_frame,k=2,users=[1,2,3,4], items=[1,2,3]
    ... ).toPandas().sort_values(["user_idx","relevance","item_idx"],
    ... ascending=[True,False,True]).reset_index(drop=True)
       user_idx  item_idx  relevance
    0         1         3   2.442027
    1         1         2   1.019667
    2         2         3   2.442027
    3         2         1   1.519667
    4         3         3   2.442027
    5         4         3   2.442027
    6         4         1   1.519667

    """

    can_predict_cold_users = True
    can_predict_cold_items = True
    item_popularity: DataFrame
    fill: float

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
            to the calculated predicted relevance
        :param seed: random seed. Provides reproducibility if fixed
        """
        # pylint: disable=super-init-not-called
        self.coef = exploration_coef
        self.sample = sample
        self.seed = seed

    @property
    def _init_args(self):
        return {
            "exploration_coef": self.coef,
            "sample": self.sample,
            "seed": self.seed,
        }

    @property
    def _dataframes(self):
        return {"item_popularity": self.item_popularity}

    def _save_model(self, path: str):
        joblib.dump({"fill": self.fill}, join(path))

    def _load_model(self, path: str):
        self.fill = joblib.load(join(path))["fill"]

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
        vals = log.select("relevance").where(
            (sf.col("relevance") != 1) & (sf.col("relevance") != 0)
        )
        if vals.count() > 0:
            raise ValueError("Relevance values in log must be 0 or 1")

        items_counts = log.groupby("item_idx").agg(
            sf.sum("relevance").alias("pos"),
            sf.count("relevance").alias("total"),
        )

        full_count = log.count()
        items_counts = items_counts.withColumn(
            "relevance",
            (
                sf.col("pos") / sf.col("total")
                + sf.sqrt(
                    sf.log(sf.lit(self.coef * full_count)) / sf.col("total")
                )
            ),
        )

        self.item_popularity = items_counts.drop("pos", "total")
        self.item_popularity.cache().count()

        self.fill = 1 + math.sqrt(math.log(self.coef * full_count))

    def _clear_cache(self):
        if hasattr(self, "item_popularity"):
            self.item_popularity.unpersist()

    def _predict_with_sampling(
        self,
        log: DataFrame,
        item_popularity: DataFrame,
        k: int,
        users: DataFrame,
    ):
        items_pd = item_popularity.withColumn(
            "probability",
            sf.col("relevance")
            / item_popularity.select(sf.sum("relevance")).first()[0],
        ).toPandas()

        seed = self.seed

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            user_idx = pandas_df["user_idx"][0]
            cnt = pandas_df["cnt"][0]

            if seed is not None:
                local_rng = default_rng(seed + user_idx)
            else:
                local_rng = default_rng()

            items_positions = local_rng.choice(
                np.arange(items_pd.shape[0]),
                size=cnt,
                p=items_pd["probability"].values,
                replace=False,
            )

            return pd.DataFrame(
                {
                    "user_idx": cnt * [user_idx],
                    "item_idx": items_pd["item_idx"].values[items_positions],
                    "relevance": items_pd["probability"].values[
                        items_positions
                    ],
                }
            )

        recs = (
            log.join(users, how="right", on="user_idx")
            .select("user_idx", "item_idx")
            .groupby("user_idx")
            .agg(sf.countDistinct("item_idx").alias("cnt"))
            .selectExpr(
                "user_idx",
                f"LEAST(cnt + {k}, {items_pd.shape[0]}) AS cnt",
            )
            .groupby("user_idx")
            .applyInPandas(grouped_map, REC_SCHEMA)
        )

        return recs

    @staticmethod
    def _calc_max_hist_len(log, users):
        max_hist_len = (
            (
                log.join(users, on="user_idx")
                .groupBy("user_idx")
                .agg(sf.countDistinct("item_idx").alias("items_count"))
            )
            .select(sf.max("items_count"))
            .collect()[0][0]
        )
        # all users have empty history
        if max_hist_len is None:
            return 0
        return max_hist_len

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

        selected_item_popularity = self.item_popularity.join(
            items,
            on="item_idx",
            how="right",
        ).fillna(value=self.fill, subset=["relevance"])

        if self.sample:
            return self._predict_with_sampling(
                log=log,
                item_popularity=selected_item_popularity,
                k=k,
                users=users,
            )

        selected_item_popularity = selected_item_popularity.withColumn(
            "rank",
            sf.row_number().over(
                Window.orderBy(
                    sf.col("relevance").desc(), sf.col("item_idx").desc()
                )
            ),
        )

        max_hist_len = (
            self._calc_max_hist_len(log, users) if filter_seen_items else 0
        )

        return users.crossJoin(
            selected_item_popularity.filter(sf.col("rank") <= k + max_hist_len)
        ).drop("rank")

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        return pairs.join(
            self.item_popularity, on="item_idx", how="left"
        ).fillna(value=self.fill, subset=["relevance"])
