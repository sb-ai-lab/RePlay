from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

from replay.data import Dataset
from replay.utils import PYSPARK_AVAILABLE, SparkDataFrame

from .base_neighbour_rec import NeighbourRec
from .extensions.ann.index_builders.base_index_builder import IndexBuilder

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf
    from pyspark.sql.window import Window


class AssociationRulesItemRec(NeighbourRec):
    """
    Item-to-item recommender based on association rules.
    Calculate pairs confidence, lift and confidence_gain defined as
    confidence(a, b)/confidence(!a, b) to get top-k associated items.
    Predict items for queries using lift, confidence or confidence_gain metrics.

    Forecasts will be based on indicators: lift, confidence or confidence_gain.
    It all depends on your choice.

    During class initialization, you can explicitly specify the metric to be used as
    a similarity for calculating the predict.

    You can change your selection before calling `.predict()` or `.predict_pairs()`
    if you set a new value for the `similarity_metric` parameter.

    Usage of ANN functionality requires only sparse indices and only one `similarity_metric`,
    defined in `__init__` will be available during inference.

    >>> import pandas as pd
    >>> from replay.data.dataset import Dataset, FeatureSchema, FeatureInfo, FeatureHint, FeatureType
    >>> from replay.utils.spark_utils import convert2spark
    >>> data_frame = pd.DataFrame({"user_id": [1, 1, 2, 3], "item_id": [1, 2, 2, 3], "rating": [2, 1, 4, 1]})
    >>> data_frame_for_predict = pd.DataFrame({"user_id": [2], "item_id": [1]})
    >>> data_frame
       user_id  item_id  rating
    0         1         1          2
    1         1         2          1
    2         2         2          4
    3         3         3          1
    >>> interactions = convert2spark(data_frame)
    >>> pred_interactions = convert2spark(data_frame_for_predict)
    >>> feature_schema = FeatureSchema(
    ...     [
    ...         FeatureInfo(
    ...             column="user_id",
    ...             feature_type=FeatureType.CATEGORICAL,
    ...             feature_hint=FeatureHint.QUERY_ID,
    ...         ),
    ...         FeatureInfo(
    ...             column="item_id",
    ...             feature_type=FeatureType.CATEGORICAL,
    ...             feature_hint=FeatureHint.ITEM_ID,
    ...         ),
    ...         FeatureInfo(
    ...             column="rating",
    ...             feature_type=FeatureType.NUMERICAL,
    ...             feature_hint=FeatureHint.RATING,
    ...         ),
    ...     ]
    ... )
    >>> train_dataset = Dataset(feature_schema, interactions)
    >>> pred_dataset = Dataset(feature_schema.subset(["user_id", "item_id"]), pred_interactions)
    >>> model = AssociationRulesItemRec(min_item_count=1, min_pair_count=0, session_column="user_id")
    >>> res = model.fit(train_dataset)
    >>> model.similarity.orderBy("item_idx_one").show()
    +------------+------------+----------+----------+----+---------------+
    |item_idx_one|item_idx_two|similarity|confidence|lift|confidence_gain|
    +------------+------------+----------+----------+----+---------------+
    |           1|           2|       1.0|       1.0| 1.5|            2.0|
    |           2|           1|       0.5|       0.5| 1.5|       Infinity|
    +------------+------------+----------+----------+----+---------------+
    >>> model.similarity_metric = "confidence"
    >>> model.predict_pairs(pred_interactions, train_dataset).show()
    +-------+-------+------+
    |user_id|item_id|rating|
    +-------+-------+------+
    |      2|      1|   0.5|
    +-------+-------+------+
    >>> model.similarity_metric = "lift"
    >>> model.predict_pairs(pred_interactions, train_dataset).show()
    +-------+-------+------+
    |user_id|item_id|rating|
    +-------+-------+------+
    |      2|      1|   1.5|
    +-------+-------+------+

    Classical model uses items co-occurrence in sessions for
    confidence, lift and confidence_gain calculation
    but rating could also be passed to the model, e.g.
    if you want to apply time smoothing and treat old sessions as less important.
    In this case all items in sessions should have the same rating.
    """

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        return {
            "features_col": None,
        }

    can_predict_item_to_item = True
    item_to_item_metrics: List[str] = ["lift", "confidence", "confidence_gain"]
    similarity: SparkDataFrame
    can_change_metric = True
    _search_space = {
        "min_item_count": {"type": "int", "args": [3, 10]},
        "min_pair_count": {"type": "int", "args": [3, 10]},
        "num_neighbours": {"type": "int", "args": [300, 2000]},
        "use_rating": {"type": "categorical", "args": [True, False]},
        "similarity_metric": {
            "type": "categorical",
            "args": ["confidence", "lift"],
        },
    }

    def __init__(
        self,
        session_column: str,
        min_item_count: int = 5,
        min_pair_count: int = 5,
        num_neighbours: Optional[int] = 1000,
        use_rating: bool = False,
        similarity_metric: str = "confidence",
        index_builder: Optional[IndexBuilder] = None,
    ) -> None:
        """
        :param session_column: name of column to group sessions.
            Items are combined by the ``user_id`` column if ``session_column`` is not defined.
        :param min_item_count: items with fewer sessions will be filtered out
        :param min_pair_count: pairs with fewer sessions will be filtered out
        :param num_neighbours: maximal number of neighbours to save for each item
        :param use_rating: flag to use rating values instead of co-occurrence count
            If true, pair rating in session is minimal rating of item in pair.
            Item rating is sum of rating in all sessions.
        :param similarity_metric: `lift` of 'confidence'
            The metric used as a similarity to calculate the prediction,
            one of [``lift``, ``confidence``, ``confidence_gain``]
        :param index_builder: `IndexBuilder` instance that adds ANN functionality.
            If not set, then ann will not be used.
        """

        self.session_column = session_column
        self.min_item_count = min_item_count
        self.min_pair_count = min_pair_count
        self.num_neighbours = num_neighbours
        self.use_rating = use_rating
        self.similarity_metric = similarity_metric
        if isinstance(index_builder, (IndexBuilder, type(None))):
            self.index_builder = index_builder
        elif isinstance(index_builder, dict):
            self.init_builder_from_dict(index_builder)

    @property
    def _init_args(self):
        return {
            "session_column": self.session_column,
            "min_item_count": self.min_item_count,
            "min_pair_count": self.min_pair_count,
            "num_neighbours": self.num_neighbours,
            "use_rating": self.use_rating,
            "similarity_metric": self.similarity_metric,
            "index_builder": self.index_builder.init_meta_as_dict() if self.index_builder else None,
        }

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        """
        1) Filter interactions items by ``min_item_count`` threshold
        2) Calculate items support, pairs confidence, lift and confidence_gain defined as
            confidence(a, b)/confidence(!a, b).
        """
        interactions = (
            dataset.interactions.withColumn(
                self.rating_column,
                sf.col(self.rating_column) if self.use_rating else sf.lit(1),
            )
            .select(self.session_column, self.item_column, self.rating_column)
            .distinct()
        )
        num_sessions = interactions.select(self.session_column).distinct().count()

        frequent_items_cached = (
            interactions.groupBy(self.item_column)
            .agg(
                sf.count(self.item_column).alias("item_count"),
                sf.sum(self.rating_column).alias("item_rating"),
            )
            .filter(sf.col("item_count") >= self.min_item_count)
            .drop("item_count")
        ).cache()

        frequent_items_interactions = interactions.join(
            frequent_items_cached.select(self.item_column), on=self.item_column
        )

        frequent_item_pairs = (
            frequent_items_interactions.withColumnRenamed(self.item_column, "antecedent")
            .withColumnRenamed(self.rating_column, "antecedent_rel")
            .join(
                frequent_items_interactions.withColumnRenamed(self.session_column, self.session_column + "_cons")
                .withColumnRenamed(self.item_column, "consequent")
                .withColumnRenamed(self.rating_column, "consequent_rel"),
                on=[
                    sf.col(self.session_column) == sf.col(self.session_column + "_cons"),
                    sf.col("antecedent") < sf.col("consequent"),
                ],
            )
            # taking minimal rating of item for pair
            .withColumn(
                self.rating_column,
                sf.least(sf.col("consequent_rel"), sf.col("antecedent_rel")),
            )
            .drop(self.session_column + "_cons", "consequent_rel", "antecedent_rel")
        )

        pairs_count = (
            frequent_item_pairs.groupBy("antecedent", "consequent")
            .agg(
                sf.count("consequent").alias("pair_count"),
                sf.sum(self.rating_column).alias("pair_rating"),
            )
            .filter(sf.col("pair_count") >= self.min_pair_count)
        ).drop("pair_count")

        pairs_metrics = pairs_count.unionByName(
            pairs_count.select(
                sf.col("consequent").alias("antecedent"),
                sf.col("antecedent").alias("consequent"),
                sf.col("pair_rating"),
            )
        )

        pairs_metrics = pairs_metrics.join(
            frequent_items_cached.withColumnRenamed("item_rating", "antecedent_rating"),
            on=[sf.col("antecedent") == sf.col(self.item_column)],
        ).drop(self.item_column)

        pairs_metrics = pairs_metrics.join(
            frequent_items_cached.withColumnRenamed("item_rating", "consequent_rating"),
            on=[sf.col("consequent") == sf.col(self.item_column)],
        ).drop(self.item_column)

        pairs_metrics = pairs_metrics.withColumn(
            "confidence",
            sf.col("pair_rating") / sf.col("antecedent_rating"),
        ).withColumn(
            "lift",
            num_sessions * sf.col("confidence") / sf.col("consequent_rating"),
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
                sf.col("consequent_rating") - sf.col("pair_rating") == 0,
                sf.lit(np.inf),
            ).otherwise(
                sf.col("confidence")
                * (num_sessions - sf.col("antecedent_rating"))
                / (sf.col("consequent_rating") - sf.col("pair_rating"))
            ),
        ).select(
            sf.col("antecedent").alias("item_idx_one"),
            sf.col("consequent").alias("item_idx_two"),
            sf.col(self.similarity_metric).alias("similarity"),
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
        items: Union[SparkDataFrame, Iterable],
        k: int,
        metric: str = "lift",
        candidates: Optional[Union[SparkDataFrame, Iterable]] = None,
    ) -> SparkDataFrame:
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
            msg = f"Select one of the valid distance metrics: {self.item_to_item_metrics}"
            raise ValueError(msg)

        return self._get_nearest_items_wrap(
            items=items,
            k=k,
            metric=metric,
            candidates=candidates,
        )

    def _get_nearest_items(
        self,
        items: SparkDataFrame,
        metric: Optional[str] = None,  # noqa: ARG002
        candidates: Optional[SparkDataFrame] = None,
    ) -> SparkDataFrame:
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
                sf.broadcast(candidates.withColumnRenamed(self.item_column, "item_idx_two")),
                on="item_idx_two",
            )

        return pairs_to_consider.join(
            sf.broadcast(items.withColumnRenamed(self.item_column, "item_idx_one")),
            on="item_idx_one",
        )

    @property
    def _dataframes(self):
        return {"similarity": self.similarity}
