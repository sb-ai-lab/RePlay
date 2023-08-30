from typing import Iterable, List, Optional, Union, Dict, Any

import numpy as np
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

from replay.models.extensions.ann.index_builders.base_index_builder import IndexBuilder
from replay.models.base_neighbour_rec import NeighbourRec


# pylint: disable=too-many-ancestors, too-many-instance-attributes
class AssociationRulesItemRec(NeighbourRec):
    """
    Item-to-item recommender based on association rules.
    Calculate pairs confidence, lift and confidence_gain defined as
    confidence(a, b)/confidence(!a, b) to get top-k associated items.
    Predict items for users using lift, confidence or confidence_gain metrics.

    Forecasts will be based on indicators: lift, confidence or confidence_gain.
    It all depends on your choice.

    During class initialization, you can explicitly specify the metric to be used as
    a similarity for calculating the predict.

    You can change your selection before calling `.predict()` or `.predict_pairs()`
    if you set a new value for the `similarity_metric` parameter.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 1, 2, 3], "item_idx": [1, 2, 2, 3], "relevance": [2, 1, 4, 1]})
    >>> data_frame_for_predict = pd.DataFrame({"user_idx": [2], "item_idx": [1]})
    >>> data_frame
       user_idx  item_idx  relevance
    0         1         1          2
    1         1         2          1
    2         2         2          4
    3         3         3          1
    >>> from replay.utils.spark_utils import convert2spark
    >>> from replay.models import AssociationRulesItemRec
    >>> data_frame = convert2spark(data_frame)
    >>> data_frame_for_predict = convert2spark(data_frame_for_predict)
    >>> model = AssociationRulesItemRec(min_item_count=1, min_pair_count=0)
    >>> res = model.fit(data_frame)
    >>> model.similarity.show()
    +------------+------------+----------+----+---------------+
    |item_idx_one|item_idx_two|confidence|lift|confidence_gain|
    +------------+------------+----------+----+---------------+
    |           1|           2|       1.0| 1.5|            2.0|
    |           2|           1|       0.5| 1.5|       Infinity|
    +------------+------------+----------+----+---------------+
    >>> model.similarity_metric = "confidence"
    >>> model.predict_pairs(data_frame_for_predict, data_frame).show()
    +--------+--------+---------+
    |user_idx|item_idx|relevance|
    +--------+--------+---------+
    |       2|       1|      0.5|
    +--------+--------+---------+
    >>> model.similarity_metric = "lift"
    >>> model.predict_pairs(data_frame_for_predict, data_frame).show()
    +--------+--------+---------+
    |user_idx|item_idx|relevance|
    +--------+--------+---------+
    |       2|       1|      1.5|
    +--------+--------+---------+

    Classical model uses items co-occurrence in sessions for
    confidence, lift and confidence_gain calculation
    but relevance could also be passed to the model, e.g.
    if you want to apply time smoothing and treat old sessions as less important.
    In this case all items in sessions should have the same relevance.
    """

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        return {
            "features_col": None,
        }

    can_predict_item_to_item = True
    item_to_item_metrics: List[str] = ["lift", "confidence", "confidence_gain"]
    similarity: DataFrame
    can_change_metric = True
    _search_space = {
        "min_item_count": {"type": "int", "args": [3, 10]},
        "min_pair_count": {"type": "int", "args": [3, 10]},
        "num_neighbours": {"type": "int", "args": [300, 2000]},
        "use_relevance": {"type": "categorical", "args": [True, False]},
        "similarity_metric": {
            "type": "categorical",
            "args": ["confidence", "lift"],
        },
    }

    # pylint: disable=too-many-arguments,
    def __init__(
        self,
        session_col: Optional[str] = None,
        min_item_count: int = 5,
        min_pair_count: int = 5,
        num_neighbours: Optional[int] = 1000,
        use_relevance: bool = False,
        similarity_metric: str = "confidence",
        index_builder: Optional[IndexBuilder] = None,
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
        :param similarity_metric: `lift` of 'confidence'
            The metric used as a similarity to calculate the prediction,
            one of [``lift``, ``confidence``, ``confidence_gain``]
        :param index_builder: `IndexBuilder` instance that adds ANN functionality.
            If not set, then ann will not be used.
        """

        self.session_col = (
            session_col if session_col is not None else "user_idx"
        )
        self.min_item_count = min_item_count
        self.min_pair_count = min_pair_count
        self.num_neighbours = num_neighbours
        self.use_relevance = use_relevance
        self.similarity_metric = similarity_metric
        if isinstance(index_builder, (IndexBuilder, type(None))):
            self.index_builder = index_builder
        elif isinstance(index_builder, dict):
            self.init_builder_from_dict(index_builder)

    @property
    def _init_args(self):
        return {
            "session_col": self.session_col,
            "min_item_count": self.min_item_count,
            "min_pair_count": self.min_pair_count,
            "num_neighbours": self.num_neighbours,
            "use_relevance": self.use_relevance,
            "similarity_metric": self.similarity_metric,
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
            "confidence_gain",
        )
        self.similarity.cache().count()
        frequent_items_cached.unpersist()

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

    def _get_nearest_items(
        self,
        items: DataFrame,
        metric: Optional[str] = None,
        candidates: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        Return metric for all available associated items filtered by `candidates`.

        :param items: items to find associated
        :param metric: `lift` of 'confidence_gain'
        :param candidates: items to consider as candidates
        :return: associated items
        """

        pairs_to_consider = self.similarity
        if candidates is not None:
            pairs_to_consider = self.similarity.join(
                sf.broadcast(
                    candidates.withColumnRenamed("item_idx", "item_idx_two")
                ),
                on="item_idx_two",
            )

        return pairs_to_consider.join(
            sf.broadcast(items.withColumnRenamed("item_idx", "item_idx_one")),
            on="item_idx_one",
        )

    @property
    def _dataframes(self):
        return {"similarity": self.similarity}
