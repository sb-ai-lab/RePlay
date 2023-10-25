from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import Recommender
from replay.data import Dataset


class UserPopRec(Recommender):
    """
    Recommends old objects from each user's personal top.
    Input is the number of interactions between users and items.

    Popularity for item :math:`i` and user :math:`u` is defined as the
    fraction of actions with item :math:`i` among all interactions of user :math:`u`:

    .. math::
        Popularity(i_u) = \\dfrac{N_iu}{N_u}

    :math:`N_iu` - number of interactions of user :math:`u` with item :math:`i`.
    :math:`N_u` - total number of interactions of user :math:`u`.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 1, 3], "item_idx": [1, 2, 3], "relevance": [2, 1, 1]})
    >>> data_frame
       user_idx  item_idx  relevance
    0         1         1          2
    1         1         2          1
    2         3         3          1

    >>> from replay.utils.spark_utils import convert2spark
    >>> data_frame = convert2spark(data_frame)
    >>> model = UserPopRec()
    >>> res = model.fit_predict(data_frame, 1, filter_seen_items=False)
    >>> model.user_item_popularity.count()
    3
    >>> res.toPandas().sort_values("user_idx", ignore_index=True)
       user_idx  item_idx  relevance
    0         1         1   0.666667
    1         3         3   1.000000
    """

    user_item_popularity: DataFrame

    @property
    def _init_args(self):
        return {}

    @property
    def _dataframes(self):
        return {"user_item_popularity": self.user_item_popularity}

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:

        user_relevance_sum = (
            dataset.interactions.groupBy(self.query_col)
            .agg(sf.sum(self.rating_col).alias("user_rel_sum"))
            .withColumnRenamed(self.query_col, "user")
            .select("user", "user_rel_sum")
        )
        self.user_item_popularity = (
            dataset.interactions.groupBy(self.query_col, self.item_col)
            .agg(sf.sum(self.rating_col).alias("user_item_rel_sum"))
            .join(
                user_relevance_sum,
                how="inner",
                on=sf.col(self.query_col) == sf.col("user"),
            )
            .select(
                self.query_col,
                self.item_col,
                (sf.col("user_item_rel_sum") / sf.col("user_rel_sum")).alias(
                    self.rating_col
                ),
            )
        )
        self.user_item_popularity.cache().count()

    def _clear_cache(self):
        if hasattr(self, "user_item_popularity"):
            self.user_item_popularity.unpersist()

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        dataset: Dataset,
        k: int,
        users: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        if filter_seen_items:
            self.logger.warning(
                "UserPopRec can't predict new items, recommendations will not be filtered"
            )

        return self.user_item_popularity.join(users, on=self.query_col).join(
            items, on=self.item_col
        )
