from replay.data import Dataset
from replay.utils import PYSPARK_AVAILABLE, SparkDataFrame

from .base_rec import Recommender

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf


class QueryPopRec(Recommender):
    """
    Recommends old objects from each query's personal top.
    Input is the number of interactions between queries and items.

    Popularity for item :math:`i` and query :math:`u` is defined as the
    fraction of actions with item :math:`i` among all interactions of query :math:`u`:

    .. math::
        Popularity(i_u) = \\dfrac{N_iu}{N_u}

    :math:`N_iu` - number of interactions of query :math:`u` with item :math:`i`.
    :math:`N_u` - total number of interactions of query :math:`u`.

    >>> import pandas as pd
    >>> from replay.data.dataset import Dataset, FeatureSchema, FeatureInfo, FeatureHint, FeatureType
    >>> from replay.utils.spark_utils import convert2spark
    >>> data_frame = pd.DataFrame({"user_id": [1, 1, 3], "item_id": [1, 2, 3], "rating": [2, 1, 1]})
    >>> data_frame
        user_id   item_id     rating
    0         1         1          2
    1         1         2          1
    2         3         3          1

    >>> interactions = convert2spark(data_frame)
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
    >>> dataset = Dataset(feature_schema, interactions)
    >>> model = QueryPopRec()
    >>> res = model.fit_predict(dataset, 1, filter_seen_items=False)
    >>> model.query_item_popularity.count()
    3
    >>> res.toPandas().sort_values("user_id", ignore_index=True)
        user_id   item_id     rating
    0         1         1   0.666667
    1         3         3   1.000000
    """

    query_item_popularity: SparkDataFrame

    @property
    def _init_args(self):
        return {}

    @property
    def _dataframes(self):
        return {"query_item_popularity": self.query_item_popularity}

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        query_rating_sum = (
            dataset.interactions.groupBy(self.query_column)
            .agg(sf.sum(self.rating_column).alias("query_rel_sum"))
            .withColumnRenamed(self.query_column, "query")
            .select("query", "query_rel_sum")
        )
        self.query_item_popularity = (
            dataset.interactions.groupBy(self.query_column, self.item_column)
            .agg(sf.sum(self.rating_column).alias("query_item_rel_sum"))
            .join(
                query_rating_sum,
                how="inner",
                on=sf.col(self.query_column) == sf.col("query"),
            )
            .select(
                self.query_column,
                self.item_column,
                (sf.col("query_item_rel_sum") / sf.col("query_rel_sum")).alias(self.rating_column),
            )
        )
        self.query_item_popularity.cache().count()

    def _clear_cache(self):
        if hasattr(self, "query_item_popularity"):
            self.query_item_popularity.unpersist()

    def _predict(
        self,
        dataset: Dataset,  # noqa: ARG002
        k: int,  # noqa: ARG002
        queries: SparkDataFrame,
        items: SparkDataFrame,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        if filter_seen_items:
            self.logger.warning("QueryPopRec can't predict new items, recommendations will not be filtered")

        return self.query_item_popularity.join(queries, on=self.query_column).join(items, on=self.item_column)
