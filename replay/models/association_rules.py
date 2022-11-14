from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import pyspark.sql.functions as sf

from pyspark.sql import DataFrame
from pyspark.sql.window import Window

from replay.models.base_rec import NeighbourRec
from replay.utils import unpersist_if_exists, to_csr
from replay.session_handler import State
from replay.constants import REC_SCHEMA


class AssociationRulesItemRec(NeighbourRec):
    """
    Item-to-item recommender based on association rules.
    Calculate pairs confidence, lift and confidence_gain defined as
    confidence(a, b)/confidence(!a, b) to get top-k associated items.

    Classical model uses items co-occurrence in sessions for
    confidence, lift and confidence_gain calculation
    but relevance could also be passed to the model, e.g.
    if you want to apply time smoothing and treat old sessions as less important.
    In this case all items in sessions should have the same relevance.
    """
    can_predict_item_to_item = True
    item_to_item_metrics: List[str] = ["lift", "confidence_gain"]
    similarity: DataFrame
    sim_column = "lift_for_pred"

    # pylint: disable=too-many-arguments,
    def __init__(
        self,
        session_col: Optional[str] = None,
        min_item_count: int = 5,
        min_pair_count: int = 5,
        num_neighbours: Optional[int] = 1000,
        use_relevance: bool = False,
    ) -> None:
        """
        :param session_col: name of column to group sessions.
            Items are combined by the ``user_id`` column if ``session_col`` is not defined.
        :param min_item_count: items with fewer sessions will be filtered out
        :param min_pair_count: pairs with fewer sessions will be filtered out
        :param num_neighbours: maximal number of neighbours to save for each item
        :param use_relevance: flag to use relevance values instead of co-occurrence count
            If true, pair relevance in session is minimal relevance of item in pair.
            Item relevance is sum of relevance in all sessions.
        """
        self.session_col = (
            session_col if session_col is not None else "user_idx"
        )
        self.min_item_count = min_item_count
        self.min_pair_count = min_pair_count
        self.num_neighbours = num_neighbours
        self.use_relevance = use_relevance

    @property
    def _init_args(self):
        return {
            "session_col": self.session_col,
            "min_item_count": self.min_item_count,
            "min_pair_count": self.min_pair_count,
            "num_neighbours": self.num_neighbours,
            "use_relevance": self.use_relevance,
        }

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        1) Filter log items by ``min_item_count`` threshold
        2) Calculate items support, pairs confidence, lift and confidence_gain defined as
            confidence(a, b)/confidence(!a, b).
        """
        log = (
            log.withColumn(
                "relevance",
                sf.col("relevance") if self.use_relevance else sf.lit(1),
            )
            .select(self.session_col, "item_idx", "relevance")
            .distinct()
        )
        num_sessions = log.select(self.session_col).distinct().count()

        frequent_items_cached = (
            log.groupBy("item_idx")
            .agg(
                sf.count("item_idx").alias("item_count"),
                sf.sum("relevance").alias("item_relevance"),
            )
            .filter(sf.col("item_count") >= self.min_item_count)
            .drop("item_count")
        ).cache()

        frequent_items_log = log.join(
            frequent_items_cached.select("item_idx"), on="item_idx"
        )

        frequent_item_pairs = (
            frequent_items_log.withColumnRenamed("item_idx", "antecedent")
            .withColumnRenamed("relevance", "antecedent_rel")
            .join(
                frequent_items_log.withColumnRenamed(
                    self.session_col, self.session_col + "_cons"
                )
                .withColumnRenamed("item_idx", "consequent")
                .withColumnRenamed("relevance", "consequent_rel"),
                on=[
                    sf.col(self.session_col)
                    == sf.col(self.session_col + "_cons"),
                    sf.col("antecedent") < sf.col("consequent"),
                ],
            )
            # taking minimal relevance of item for pair
            .withColumn(
                "relevance",
                sf.least(sf.col("consequent_rel"), sf.col("antecedent_rel")),
            )
            .drop(
                self.session_col + "_cons", "consequent_rel", "antecedent_rel"
            )
        )

        pairs_count = (
            frequent_item_pairs.groupBy("antecedent", "consequent")
            .agg(
                sf.count("consequent").alias("pair_count"),
                sf.sum("relevance").alias("pair_relevance"),
            )
            .filter(sf.col("pair_count") >= self.min_pair_count)
        ).drop("pair_count")

        pairs_metrics = pairs_count.unionByName(
            pairs_count.select(
                sf.col("consequent").alias("antecedent"),
                sf.col("antecedent").alias("consequent"),
                sf.col("pair_relevance"),
            )
        )

        pairs_metrics = pairs_metrics.join(
            frequent_items_cached.withColumnRenamed(
                "item_relevance", "antecedent_relevance"
            ),
            on=[sf.col("antecedent") == sf.col("item_idx")],
        ).drop("item_idx")

        pairs_metrics = pairs_metrics.join(
            frequent_items_cached.withColumnRenamed(
                "item_relevance", "consequent_relevance"
            ),
            on=[sf.col("consequent") == sf.col("item_idx")],
        ).drop("item_idx")

        pairs_metrics = pairs_metrics.withColumn(
            "confidence",
            sf.col("pair_relevance") / sf.col("antecedent_relevance"),
        ).withColumn(
            "lift",
            num_sessions
            * sf.col("confidence")
            / sf.col("consequent_relevance"),
        )

        if self.num_neighbours is not None:
            pairs_metrics = (
                pairs_metrics.withColumn(
                    "similarity_order",
                    sf.row_number().over(
                        Window.partitionBy("antecedent").orderBy(
                            sf.col("lift").desc(),
                            sf.col("consequent").desc(),
                        )
                    ),
                )
                .filter(sf.col("similarity_order") <= self.num_neighbours)
                .drop("similarity_order")
            )

        self.similarity = pairs_metrics.withColumn(
            "confidence_gain",
            sf.when(
                sf.col("consequent_relevance") - sf.col("pair_relevance") == 0,
                sf.lit(np.inf),
            ).otherwise(
                sf.col("confidence")
                * (num_sessions - sf.col("antecedent_relevance"))
                / (sf.col("consequent_relevance") - sf.col("pair_relevance"))
            ),
        ).select(
            sf.col("antecedent").alias("item_idx_one"),
            sf.col("consequent").alias("item_idx_two"),
            "confidence",
            "lift",
            sf.log("lift").alias(self.sim_column),
            "confidence_gain",
        )
        self.similarity.cache().count()
        frequent_items_cached.unpersist()

    # def get_sim_csr(self):
    #     sim_log = (
    #         self.similarity
    #             .select(sf.col("antecedent").alias("user_idx"),
    #                     sf.col("consequent").alias("item_idx"),
    #                     sf.log("lift").alias("relevance"))
    #     )
    #
    #     max_idx_from_sim = sim_log.select(sf.max("item_idx")).first()[0]
    #     max_idx_from_log = self.fit_items.select(sf.max("item_idx")).first()[0]
    #
    #     if max_idx_from_sim < max_idx_from_log:
    #         spark = State().session
    #         sim_log = sim_log.union(
    #             spark.createDataFrame(
    #                 data=[
    #                     [max_idx_from_log, max_idx_from_log, 0.]
    #                 ],
    #                 schema=sim_log.schema,
    #             )
    #         )
    #
    #     return to_csr(sim_log)
    #
    #
    # # pylint: disable=too-many-arguments
    # def _predict(
    #     self,
    #     log: DataFrame,
    #     k: int,
    #     users: DataFrame,
    #     items: DataFrame,
    #     user_features: Optional[DataFrame] = None,
    #     item_features: Optional[DataFrame] = None,
    #     filter_seen_items: bool = True,
    # ) -> DataFrame:
    #
    #     def predict_top_by_user(pandas_df):
    #         user_idx = int(pandas_df["user_idx"].iloc[0])
    #         uniq = pandas_df["item_idx"].unique()
    #         mul = csr_matrix((pandas_df["relevance"], ([0] * (uniq.shape[0]), uniq)), shape=(1, sim_csr.shape[0]))
    #         res = mul @ sim_csr
    #         res_arr = res.toarray()[0]
    #         return pd.DataFrame(
    #             {
    #                 "user_idx": [user_idx] * np.argsort(res_arr).shape[0],
    #                 "item_idx": np.argsort(res_arr)[::-1],
    #                 "relevance": np.sort(res_arr)[::-1],
    #             }
    #         )
    #
    #     sim_csr = self.get_sim_csr()
    #
    #     return (
    #             log
    #             .join(users, on='user_idx')
    #             .select("user_idx", "item_idx", "relevance")
    #             .groupby("user_idx")
    #             .applyInPandas(predict_top_by_user, REC_SCHEMA)
    #     ).join(items, on='item_idx')
    #
    #
    # def _predict_pairs(
    #     self,
    #     pairs: DataFrame,
    #     log: Optional[DataFrame] = None,
    #     user_features: Optional[DataFrame] = None,
    #     item_features: Optional[DataFrame] = None,
    # ) -> DataFrame:
    #
    #     if log is None:
    #         raise ValueError(
    #             "log is not provided, but it is required for prediction"
    #         )
    #
    #     def predict_pairs(pandas_df):
    #         user_idx = int(pandas_df["user_idx"].iloc[0])
    #         uniq = pandas_df["item_idx"].unique()
    #         mul = csr_matrix((pandas_df["relevance"], ([0] * (uniq.shape[0]), uniq)), shape=(1, sim_csr.shape[0]))
    #         res = mul @ sim_csr
    #         _, idx_for_res = pairs_csr[user_idx].nonzero()
    #
    #         res_arr = res.toarray()[0][idx_for_res]
    #         return pd.DataFrame(
    #             {
    #                 "user_idx": [user_idx] * idx_for_res.shape[0],
    #                 "item_idx": idx_for_res,
    #                 "relevance": res_arr,
    #             }
    #         )
    #
    #     sim_csr = self.get_sim_csr()
    #     pairs_csr = to_csr(pairs.withColumn("relevance", sf.lit(1)))
    #     return (
    #             log.join(pairs.select("user_idx").distinct(), on="user_idx")
    #             .select("user_idx", "item_idx", "relevance")
    #             .groupby("user_idx")
    #             .applyInPandas(predict_pairs, REC_SCHEMA)
    #     )


    @property
    def get_similarity(self):
        """
        Return matrix with calculated confidence, lift and confidence gain.
        :return: association rules measures calculated during ``fit`` stage
        """
        return self.similarity

    def get_nearest_items(
        self,
        items: Union[DataFrame, Iterable],
        k: int,
        metric: str = "lift",
        candidates: Optional[Union[DataFrame, Iterable]] = None,
    ) -> DataFrame:
        """
        Get k most similar items be the `metric` for each of the `items`.

        :param items: spark dataframe or list of item ids to find neighbors
        :param k: number of neighbors
        :param metric: `lift` of 'confidence_gain'
        :param candidates: spark dataframe or list of items
            to consider as similar, e.g. popular/new items. If None,
            all items presented during model training are used.
        :return: dataframe with the most similar items an distance,
            where bigger value means greater similarity.
            spark-dataframe with columns ``[item_id, neighbour_item_id, similarity]``
        """
        if metric not in self.item_to_item_metrics:
            raise ValueError(
                f"Select one of the valid distance metrics: "
                f"{self.item_to_item_metrics}"
            )

        return self._get_nearest_items_wrap(
            items=items,
            k=k,
            metric=metric,
            candidates=candidates,
        )

    def _clear_cache(self):
        if hasattr(self, "similarity"):
            unpersist_if_exists(self.similarity)

    @property
    def _dataframes(self):
        return {"similarity": self.similarity}
