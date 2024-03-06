from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import polars as pl

from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, SparkDataFrame, PolarsDataFrame

from .base_metric import (
    Metric,
    MetricsDataFrameLike,
    MetricsMeanReturnType,
    MetricsPerUserReturnType,
    MetricsReturnType,
)
from .descriptors import CalculationDescriptor, Mean

if PYSPARK_AVAILABLE:
    from pyspark.sql import Window
    from pyspark.sql import functions as F


# pylint: disable=too-few-public-methods
class CategoricalDiversity(Metric):
    """
    Metric calculation is as follows:

    * take ``K`` recommendations with the biggest ``score`` for each ``user_id``
    * count the number of distinct ``category_id`` in these recommendations / ``K``
    * average this number for all users

    >>> category_recommendations
       query_id  category_id  rating
    0         1            3    0.6
    1         1            7    0.5
    2         1           10    0.4
    3         1           11    0.3
    4         1            2    0.2
    5         2            5    0.6
    6         2            8    0.5
    7         2           11    0.4
    8         2            1    0.3
    9         2            3    0.2
    10        3            4    1.0
    11        3            9    0.5
    12        3            2    0.1
    >>> from replay.metrics import Median, ConfidenceInterval, PerUser
    >>> CategoricalDiversity([3, 5])(category_recommendations)
    {'CategoricalDiversity@3': 1.0, 'CategoricalDiversity@5': 0.8666666666666667}
    >>> CategoricalDiversity([3, 5], mode=PerUser())(category_recommendations)
    {'CategoricalDiversity-PerUser@3': {1: 1.0, 2: 1.0, 3: 1.0},
     'CategoricalDiversity-PerUser@5': {1: 1.0, 2: 1.0, 3: 0.6}}
    >>> CategoricalDiversity([3, 5], mode=Median())(category_recommendations)
    {'CategoricalDiversity-Median@3': 1.0,
     'CategoricalDiversity-Median@5': 1.0}
    >>> CategoricalDiversity([3, 5], mode=ConfidenceInterval(alpha=0.95))(category_recommendations)
    {'CategoricalDiversity-ConfidenceInterval@3': 0.0,
     'CategoricalDiversity-ConfidenceInterval@5': 0.2613285312720073}
    <BLANKLINE>
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        topk: Union[List, int],
        query_column: str = "query_id",
        category_column: str = "category_id",
        rating_column: str = "rating",
        mode: CalculationDescriptor = Mean(),
    ):
        """
        :param topk: (list or int): Consider the highest k scores in the ranking.
        :param user_column: (str): The name of the user column.
        :param category_column: (str): The name of the category column.
        :param score_column: (str): The name of the score column.
        :param mode: (CalculationDescriptor): class for calculating aggregation metrics.
            Default: ``Mean``.
        """
        super().__init__(
            topk=topk,
            query_column=query_column,
            rating_column=rating_column,
            mode=mode,
        )
        self.category_column = category_column

    def __call__(self, recommendations: MetricsDataFrameLike) -> MetricsReturnType:
        """
        Compute metric.

        :param recommendations: (PySpark DataFrame or Polars DataFrame or Pandas DataFrame or dict):
            model predictions.
            If DataFrame then it must contains user, category and score columns.
            If dict then key represents user_ids, value represents list of tuple(category, score).

        :return: metric values
        """
        if isinstance(recommendations, SparkDataFrame):
            return self._spark_call(recommendations)
        if isinstance(recommendations, PolarsDataFrame):
            return self._polars_call(recommendations)
        is_pandas = isinstance(recommendations, PandasDataFrame)
        recommendations = (
            self._convert_pandas_to_dict_with_score(recommendations)
            if is_pandas
            else self._convert_dict_to_dict_with_score(recommendations)
        )
        precalculated_answer = self._precalculate_unique_cats(recommendations)
        return self._dict_call(precalculated_answer)

    # pylint: disable=arguments-differ
    def _get_enriched_recommendations(
        self, recommendations: Union[PolarsDataFrame, SparkDataFrame],
    ) -> Union[PolarsDataFrame, SparkDataFrame]:
        if isinstance(recommendations, SparkDataFrame):
            return self._get_enriched_recommendations_spark(recommendations)
        else:
            return self._get_enriched_recommendations_polars(recommendations)

    # pylint: disable=arguments-differ
    def _get_enriched_recommendations_spark(
        self, recommendations: SparkDataFrame
    ) -> SparkDataFrame:
        window = Window.partitionBy(self.query_column).orderBy(
            F.col(self.rating_column).desc()
        )
        sorted_by_score_recommendations = recommendations.withColumn(
            "rank", F.row_number().over(window)
        )
        return sorted_by_score_recommendations

    # pylint: disable=arguments-differ
    def _get_enriched_recommendations_polars(
        self, recommendations: PolarsDataFrame
    ) -> PolarsDataFrame:
        sorted_by_score_recommendations = recommendations.select(
            pl.all().sort_by(self.rating_column, descending=True).over(self.query_column)
        )
        sorted_by_score_recommendations = sorted_by_score_recommendations.with_columns(
            sorted_by_score_recommendations.select(
                pl.col(self.query_column).cum_count().over(self.query_column).alias("rank")
            )
        )
        return sorted_by_score_recommendations

    def _spark_compute_per_user(self, recs: SparkDataFrame) -> MetricsPerUserReturnType:
        distribution_per_user = defaultdict(list)
        for k in self.topk:
            filtered_recs = recs.filter(F.col("rank") <= k)
            aggreagated_by_user = filtered_recs.groupBy(self.query_column).agg(
                F.countDistinct(self.category_column)
            )
            aggreagated_by_user_dict = (
                aggreagated_by_user.rdd.collectAsMap()
            )  # type:ignore
            for user, metric in aggreagated_by_user_dict.items():
                distribution_per_user[user].append(metric / k)
        return self._aggregate_results_per_user(dict(distribution_per_user))

    def _polars_compute_per_user(self, recs: PolarsDataFrame) -> MetricsPerUserReturnType:
        distribution_per_user = defaultdict(list)
        for k in self.topk:
            filtered_recs = recs.filter(pl.col("rank") <= k)
            aggreagated_by_user = filtered_recs.group_by(self.query_column).agg(
                pl.col(self.category_column).n_unique()
            )
            aggreagated_by_user_dict = (
                dict(aggreagated_by_user.iter_rows())
            )  # type:ignore
            for user, metric in aggreagated_by_user_dict.items():
                distribution_per_user[user].append(metric / k)
        return self._aggregate_results_per_user(dict(distribution_per_user))

    def _spark_compute_agg(self, recs: SparkDataFrame) -> MetricsMeanReturnType:
        metrics = []
        for k in self.topk:
            filtered_recs = recs.filter(F.col("rank") <= k)
            aggregated_by_user = (
                filtered_recs.groupBy(self.query_column)
                .agg(F.countDistinct(self.category_column))
                .drop(self.query_column)
            )
            metrics.append(self._mode.spark(aggregated_by_user) / k)
        return self._aggregate_results(metrics)

    def _polars_compute_agg(self, recs: PolarsDataFrame) -> MetricsMeanReturnType:
        metrics = []
        for k in self.topk:
            filtered_recs = recs.filter(pl.col("rank") <= k)
            aggregated_by_user = (
                filtered_recs.group_by(self.query_column)
                .agg(pl.col(self.category_column).n_unique())
                .drop(self.query_column)
            )
            metrics.append(self._mode.cpu(aggregated_by_user) / k)
        return self._aggregate_results(metrics)

    # pylint: disable=arguments-differ
    def _spark_call(self, recommendations: SparkDataFrame) -> MetricsReturnType:
        """
        Implementation for Pyspark DataFrame.
        """
        recs = self._get_enriched_recommendations(recommendations)
        if self._mode.__name__ == "PerUser":
            return self._spark_compute_per_user(recs)
        return self._spark_compute_agg(recs)

    # pylint: disable=arguments-differ
    def _polars_call(self, recommendations: PolarsDataFrame) -> MetricsReturnType:
        """
        Implementation for Polars DataFrame.
        """
        recs = self._get_enriched_recommendations(recommendations)
        if self._mode.__name__ == "PerUser":
            return self._polars_compute_per_user(recs)
        return self._polars_compute_agg(recs)

    def _convert_pandas_to_dict_with_score(self, data: PandasDataFrame) -> Dict:
        return (
            data.sort_values(by=self.rating_column, ascending=False)
            .groupby(self.query_column)[self.category_column]
            .apply(list)
            .to_dict()
        )

    # pylint: disable=no-self-use
    def _precalculate_unique_cats(self, recommendations: Dict) -> Dict:
        """
        Precalculate unique categories for each prefix for each user.
        """
        answer = {}
        for user, cats in recommendations.items():
            unique = set()
            unique_len = []
            for cat in cats:
                unique.add(cat)
                unique_len.append(len(unique))
            answer[user] = unique_len
        return answer

    # pylint: disable=arguments-renamed,arguments-differ
    def _dict_compute_per_user(
        self, precalculated_answer: Dict
    ) -> MetricsPerUserReturnType:  # type:ignore
        distribution_per_user = defaultdict(list)
        for k in self.topk:
            for user, unique_cats in precalculated_answer.items():
                distribution_per_user[user].append(
                    unique_cats[min(len(unique_cats), k) - 1] / k
                )
        return self._aggregate_results_per_user(distribution_per_user)

    # pylint: disable=arguments-renamed
    def _dict_compute_mean(
        self, precalculated_answer: Dict
    ) -> MetricsMeanReturnType:  # type:ignore
        distribution_list = []
        for _, unique_cats in precalculated_answer.items():
            metrics_per_user = []
            for k in self.topk:
                metric = unique_cats[min(len(unique_cats), k) - 1] / k
                metrics_per_user.append(metric)
            distribution_list.append(metrics_per_user)

        distribution = np.stack(distribution_list)
        assert distribution.shape[1] == len(self.topk)
        metrics = []
        for k in range(distribution.shape[1]):
            metrics.append(self._mode.cpu(distribution[:, k]))
        return self._aggregate_results(metrics)

    # pylint: disable=arguments-differ
    def _dict_call(self, precalculated_answer: Dict) -> MetricsReturnType:
        """
        Calculating metrics in dict format.
        """
        if self._mode.__name__ == "PerUser":
            return self._dict_compute_per_user(precalculated_answer)
        return self._dict_compute_mean(precalculated_answer)

    @staticmethod
    def _get_metric_value_by_user(
        ks: List[int], *args: List
    ) -> List[float]:  # pragma: no cover
        pass
