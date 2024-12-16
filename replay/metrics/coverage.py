import functools
import operator
from typing import Dict, List, Union

import polars as pl

from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, PolarsDataFrame, SparkDataFrame

from .base_metric import Metric, MetricsDataFrameLike, MetricsMeanReturnType, MetricsReturnType

if PYSPARK_AVAILABLE:
    from pyspark.sql import (
        Window,
        functions as sf,
    )


class Coverage(Metric):
    """
    Metric calculation is as follows:

    * take ``K`` recommendations with the biggest ``score`` for each ``user_id``
    * count the number of distinct ``item_id`` in these recommendations
    * divide it by the number of distinct items in train dataset, provided to metric call

    >>> recommendations
       query_id  item_id  rating
    0         1        3    0.6
    1         1        7    0.5
    2         1       10    0.4
    3         1       11    0.3
    4         1        2    0.2
    5         2        5    0.6
    6         2        8    0.5
    7         2       11    0.4
    8         2        1    0.3
    9         2        3    0.2
    10        3        4    1.0
    11        3        9    0.5
    12        3        2    0.1
    >>> train
       query_id  item_id
    0         1        5
    1         1        6
    2         1        8
    3         1        9
    4         1        2
    5         2        5
    6         2        8
    7         2       11
    8         2        1
    9         2        3
    10        3        4
    11        3        9
    12        3        2
    >>> Coverage(2)(recommendations, train)
    {'Coverage@2': 0.5555555555555556}
    <BLANKLINE>
    """

    def __init__(
        self,
        topk: Union[List, int],
        query_column: str = "query_id",
        item_column: str = "item_id",
        rating_column: str = "rating",
        allow_caching: bool = True,
    ) -> None:
        """
        :param topk: (list or int): Consider the highest k scores in the ranking.
        :param query_column: (str): The name of the user column.
        :param item_column: (str): The name of the item column.
        :param rating_column: (str): The name of the score column.
        :param allow_caching: (bool): The flag for using caching to optimize calculations.
            Default: ``True``.
        """
        super().__init__(
            topk=topk,
            query_column=query_column,
            item_column=item_column,
            rating_column=rating_column,
        )
        self._allow_caching = allow_caching

    def _get_enriched_recommendations(
        self,
        recommendations: Union[PolarsDataFrame, SparkDataFrame],
    ) -> Union[PolarsDataFrame, SparkDataFrame]:
        if isinstance(recommendations, SparkDataFrame):
            return self._get_enriched_recommendations_spark(recommendations)
        else:
            return self._get_enriched_recommendations_polars(recommendations)

    def _get_enriched_recommendations_spark(self, recommendations: SparkDataFrame) -> SparkDataFrame:
        window = Window.partitionBy(self.query_column).orderBy(sf.col(self.rating_column).desc())
        sorted_by_score_recommendations = recommendations.withColumn("rank", sf.row_number().over(window))
        grouped_recs = (
            sorted_by_score_recommendations.select(self.item_column, "rank")
            .groupBy(self.item_column)
            .agg(sf.min("rank").alias("best_position"))
        )
        return grouped_recs

    def _get_enriched_recommendations_polars(self, recommendations: PolarsDataFrame) -> PolarsDataFrame:
        sorted_by_score_recommendations = recommendations.select(
            pl.all().sort_by(self.rating_column, descending=True).over(self.query_column)
        )
        sorted_by_score_recommendations = sorted_by_score_recommendations.with_columns(
            sorted_by_score_recommendations.select(
                pl.col(self.query_column).cum_count().over(self.query_column).alias("rank")
            )
        )
        grouped_recs = (
            sorted_by_score_recommendations.select(self.item_column, "rank")
            .group_by(self.item_column)
            .agg(pl.col("rank").min().alias("best_position"))
        )
        return grouped_recs

    def _spark_compute(self, recs: SparkDataFrame, train: SparkDataFrame) -> MetricsMeanReturnType:
        """
        Calculating metrics for PySpark DataFrame.
        """
        if self._allow_caching:
            recs.cache()

        item_count = train.select(self.item_column).distinct().count()

        metrics = []
        for k in self.topk:
            res = (
                recs.filter(sf.col("best_position") <= k)
                .select(self.item_column)
                .distinct()
                .join(train.select(self.item_column).distinct(), on=self.item_column)
                .count()
                / item_count
            )
            metrics.append(res)

        if self._allow_caching:
            recs.unpersist()

        return self._aggregate_results(metrics)

    def _polars_compute(self, recs: PolarsDataFrame, train: PolarsDataFrame) -> MetricsMeanReturnType:
        """
        Calculating metrics for Polars DataFrame.
        """
        item_count = train.n_unique(self.item_column)

        metrics = []
        for k in self.topk:
            res = (
                recs.filter(pl.col("best_position") <= k)
                .select(self.item_column)
                .unique()
                .join(train.select(self.item_column).unique(), on=self.item_column)
                .count()
                / item_count
            ).rows()[0][0]
            metrics.append(res)

        return self._aggregate_results(metrics)

    def _spark_call(self, recommendations: SparkDataFrame, train: SparkDataFrame) -> MetricsReturnType:
        """
        Implementation for Pyspark DataFrame.
        """
        recs = self._get_enriched_recommendations(recommendations)
        return self._spark_compute(recs, train)

    def _polars_call(self, recommendations: PolarsDataFrame, train: PolarsDataFrame) -> MetricsReturnType:
        """
        Implementation for Polars DataFrame.
        """
        recs = self._get_enriched_recommendations(recommendations)
        return self._polars_compute(recs, train)

    def _dict_call(self, recommendations: Dict, train: Dict) -> MetricsReturnType:
        """
        Calculating metrics in dict format.
        """
        train_items = set(functools.reduce(operator.iconcat, train.values(), []))

        len_train_items = len(train_items)
        metrics = []
        for k in self.topk:
            pred_items = set()
            for items in recommendations.values():
                for item in items[:k]:
                    pred_items.add(item)
            metrics.append(len(pred_items & train_items) / len_train_items)
        return self._aggregate_results(metrics)

    def __call__(
        self,
        recommendations: MetricsDataFrameLike,
        train: MetricsDataFrameLike,
    ) -> MetricsReturnType:
        """
        Compute metric.

        :param recommendations: (PySpark DataFrame or Polars DataFrame or Pandas DataFrame or dict):
            model predictions.
            If DataFrame then it must contains user, item and score columns.
            If dict then key represents user_ids, value represents list of tuple(item_id, score).
        :param train: (PySpark DataFrame or Polars DataFrame or Pandas DataFrame or dict):
            train data.
            If DataFrame then it must contains user and item columns.
            If dict then key represents user_ids, value represents list of item_ids.

        :return: metric values
        """
        self._check_dataframes_equal_types(recommendations, train)
        if isinstance(recommendations, SparkDataFrame):
            self._check_duplicates_spark(recommendations)
            assert isinstance(train, SparkDataFrame)
            return self._spark_call(recommendations, train)
        if isinstance(recommendations, PolarsDataFrame):
            self._check_duplicates_polars(recommendations)
            assert isinstance(train, PolarsDataFrame)
            return self._polars_call(recommendations, train)
        is_pandas = isinstance(recommendations, PandasDataFrame)
        recommendations = (
            self._convert_pandas_to_dict_with_score(recommendations)
            if is_pandas
            else self._convert_dict_to_dict_with_score(recommendations)
        )
        self._check_duplicates_dict(recommendations)
        train = self._convert_pandas_to_dict_without_score(train) if is_pandas else train
        assert isinstance(train, dict)
        return self._dict_call(recommendations, train)

    @staticmethod
    def _get_metric_value_by_user(ks, *args) -> List[float]:  # pragma: no cover
        pass
