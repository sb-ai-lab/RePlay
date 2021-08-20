from typing import Optional

import numpy as np
import pyspark.sql.functions as sf

from pyspark.sql import DataFrame
from pyspark.sql.window import Window

from replay.models.base_rec import Recommender
from replay.utils import unpersist_if_exists


class AssociationRulesItemRec(Recommender):
    """
    Item-to-item recommender based on association rules.
    Calculate pairs confidence, lift and confidence_gain defined as
        confidence(a, b)/confidence(!a, b) to get top-k associated items.
    """

    can_predict_item_to_item = True

    frequent_items: DataFrame
    num_sessions: int
    pairs_metrics: DataFrame

    def __init__(
        self,
        session_col_name: Optional[str] = None,
        min_item_count: int = 5,
        min_pair_count: int = 5,
        num_neighbours: Optional[int] = 1000,
    ) -> None:
        """
        :param session_col_name: name of column to group sessions.
            Items are combined by the ``user_id`` column if ``session_col_name`` is not defined.
        :param min_item_count items with fewer number of sessions will be filtered out
        :param min_item_count pairs with fewer number of sessions will be filtered out
        :param num_neighbours maximal number of neighbours to save for each item
        """
        self.session_col_name = (
            session_col_name if session_col_name is not None else "user_idx"
        )
        self.min_item_count = min_item_count
        self.min_pair_count = min_pair_count
        self.num_neighbours = num_neighbours

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:

        """
        1) Filter log to leave items present in log more then ``min_item_count`` threshold.
        2) Calculate items support, pairs confidence, lift and confidence_gain defined as
            confidence(a, b)/confidence(!a, b).
        """
        log = log.select(self.session_col_name, "item_idx").distinct()

        self.num_sessions = (
            log.select(self.session_col_name).distinct().count()
        )

        self.frequent_items = (
            log.groupBy("item_idx")
            .agg(sf.count("item_idx").alias("item_count"))
            .filter(sf.col("item_count") >= self.min_item_count)
        ).cache()

        frequent_items_log = log.join(
            sf.broadcast(self.frequent_items.select("item_idx")), on="item_idx"
        )

        frequent_item_pairs = (
            frequent_items_log.withColumnRenamed("item_idx", "antecedent")
            .join(
                frequent_items_log.withColumnRenamed(
                    self.session_col_name, self.session_col_name + "_cons"
                ).withColumnRenamed("item_idx", "consequent"),
                on=[
                    sf.col(self.session_col_name)
                    == sf.col(self.session_col_name + "_cons"),
                    sf.col("antecedent") < sf.col("consequent"),
                ],
            )
            .drop(self.session_col_name + "_cons")
        )
        pairs_count = frequent_item_pairs.groupBy(
            "antecedent", "consequent"
        ).agg(sf.count("consequent").alias("pair_count"))

        pairs_count = pairs_count.filter(
            sf.col("pair_count") >= self.min_pair_count
        )

        pairs_metrics = pairs_count.unionByName(
            pairs_count.select(
                sf.col("consequent").alias("antecedent"),
                sf.col("antecedent").alias("consequent"),
                sf.col("pair_count"),
            )
        )
        pairs_metrics = pairs_metrics.join(
            self.frequent_items.withColumnRenamed(
                "item_count", "antecedent_count"
            ),
            on=[sf.col("antecedent") == sf.col("item_idx")],
        ).drop("item_idx")

        pairs_metrics = pairs_metrics.join(
            self.frequent_items.withColumnRenamed(
                "item_count", "consequent_count"
            ),
            on=[sf.col("consequent") == sf.col("item_idx")],
        ).drop("item_idx")

        pairs_metrics = pairs_metrics.withColumn(
            "confidence", sf.col("pair_count") / sf.col("antecedent_count")
        )

        pairs_metrics = pairs_metrics.withColumn(
            "lift",
            self.num_sessions
            * sf.col("confidence")
            / sf.col("consequent_count"),
        )

        if self.num_neighbours is not None:
            pairs_metrics = (
                pairs_metrics.withColumn(
                    "similarity_order",
                    sf.row_number().over(
                        Window.partitionBy("antecedent").orderBy(
                            sf.col("lift").desc(), sf.col("consequent").desc(),
                        )
                    ),
                )
                .filter(sf.col("similarity_order") <= self.num_neighbours)
                .drop("similarity_order")
            )

        self.pairs_metrics = pairs_metrics.withColumn(
            "confidence_gain",
            sf.when(
                sf.col("consequent_count") - sf.col("pair_count") == 0,
                sf.lit(np.inf),
            ).otherwise(
                sf.col("confidence")
                * (self.num_sessions - sf.col("antecedent_count"))
                / (sf.col("consequent_count") - sf.col("pair_count"))
            ),
        )

        # self.pairs_metrics = self.pairs_metrics.select(
        #     "antecedent", "consequent", "confidence", "lift", "confidence_gain"
        # ).cache()

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
    ) -> None:
        raise NotImplementedError(
            "item-to-user predict is not implemented for {}, "
            "use get_nearest_items method to get item-to-item recommendations".format(
                self.__str__()
            )
        )

    def _get_nearest_items(
        self,
        items: DataFrame,
        metric: Optional[str] = None,
        items_to_consider: Optional[DataFrame] = None,
    ) -> Optional[DataFrame]:
        """
        For each item return top-k items with the highest values
        of chosen metric (`lift` of `confidence_gain`) from ``items_to_consider``
        :param items: items to find associated
        :param metric: `lift` of 'confidence_gain'
        :param items_to_consider: items to consider as candidates
        :return: associated items
        """

        pairs_to_consider = self.pairs_metrics
        if items_to_consider is not None:
            pairs_to_consider = self.pairs_metrics.join(
                sf.broadcast(
                    items_to_consider.withColumnRenamed(
                        "item_idx", "consequent"
                    )
                ),
                on="consequent",
            )

        return (
            pairs_to_consider.withColumnRenamed("antecedent", "item_id_one")
            .withColumnRenamed("consequent", "item_id_two")
            .join(
                sf.broadcast(
                    items.withColumnRenamed("item_idx", "item_id_one")
                ),
                on="item_id_one",
            )
        )

    def _clear_cache(self):
        if hasattr(self, "pairs_metrics"):
            unpersist_if_exists(self.pairs_metrics)
        if hasattr(self, "frequent_items"):
            unpersist_if_exists(self.frequent_items)
