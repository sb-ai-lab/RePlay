import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import polars as pl

from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, PandasDataFrame, PolarsDataFrame, SparkDataFrame

from .descriptors import CalculationDescriptor, Mean

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf
    from pyspark.sql.types import ArrayType, DoubleType, StructType


MetricsDataFrameLike = Union[DataFrameLike, Dict]
MetricsMeanReturnType = Mapping[str, float]
MetricsPerUserReturnType = Mapping[str, Mapping[Any, float]]
MetricsReturnType = Union[MetricsMeanReturnType, MetricsPerUserReturnType]


class MetricDuplicatesWarning(Warning):
    """Recommendations contain duplicates"""


class Metric(ABC):
    """Base metric class"""

    def __init__(
        self,
        topk: Union[List[int], int],
        query_column: str = "query_id",
        item_column: str = "item_id",
        rating_column: str = "rating",
        mode: CalculationDescriptor = Mean(),
    ) -> None:
        """
        :param topk: (list or int): Consider the highest k scores in the ranking.
        :param query_column: (str): The name of the user column.
        :param item_column: (str): The name of the item column.
        :param rating_column: (str): The name of the score column.
        :param mode: (CalculationDescriptor): class for calculating aggregation metrics.
            Default: ``Mean``.
        """
        if isinstance(topk, list):
            for item in topk:
                if not isinstance(item, int):
                    msg = f"{item} is not int"
                    raise ValueError(msg)
        elif isinstance(topk, int):
            topk = [topk]
        else:
            msg = "topk not list or int"
            raise ValueError(msg)
        self.topk = sorted(topk)
        self.query_column = query_column
        self.item_column = item_column
        self.rating_column = rating_column
        self._mode = mode

    @property
    def __name__(self) -> str:
        mode_name = self._mode.__name__
        return str(type(self).__name__) + (f"-{mode_name}" if mode_name != "Mean" else "")

    def _check_dataframes_equal_types(
        self,
        recommendations: MetricsDataFrameLike,
        ground_truth: MetricsDataFrameLike,
    ) -> None:
        """
        Types of all data frames must be the same.
        """
        if not isinstance(recommendations, type(ground_truth)):
            msg = "All given data frames must have the same type"
            raise ValueError(msg)

    def _duplicate_warn(self):
        warnings.warn(
            "The recommendations contain duplicated users and items.The metrics may be higher than the actual ones.",
            MetricDuplicatesWarning,
        )

    def _check_duplicates_spark(self, recommendations: SparkDataFrame) -> None:
        duplicates_count = (
            recommendations.groupBy(self.query_column, self.item_column).count().filter("count >= 2").count()
        )
        if duplicates_count:
            self._duplicate_warn()

    def _check_duplicates_dict(self, recommendations: Dict) -> None:
        for items in recommendations.values():
            items_set = set(items)
            if len(items) != len(items_set):
                self._duplicate_warn()
                return

    def _check_duplicates_polars(self, recommendations: PolarsDataFrame) -> None:
        duplicates_count = recommendations.group_by(self.query_column, self.item_column).len().filter(pl.col("len") > 1)
        if not duplicates_count.is_empty():
            self._duplicate_warn()

    def __call__(
        self,
        recommendations: MetricsDataFrameLike,
        ground_truth: MetricsDataFrameLike,
    ) -> MetricsReturnType:
        """
        Compute metric.

        :param recommendations: (PySpark DataFrame or Polars DataFrame or Pandas DataFrame or dict): model predictions.
            If DataFrame then it must contains user, item and score columns.
            If dict then key represents user_ids, value represents list of tuple(item_id, score).
        :param ground_truth: (PySpark DataFrame or Polars DataFrame or Pandas DataFrame or dict):
            test data.
            If DataFrame then it must contains user and item columns.
            If dict then key represents user_ids, value represents list of item_ids.

        :return: metric values
        """
        self._check_dataframes_equal_types(recommendations, ground_truth)
        if isinstance(recommendations, SparkDataFrame):
            self._check_duplicates_spark(recommendations)
            assert isinstance(ground_truth, SparkDataFrame)
            return self._spark_call(recommendations, ground_truth)
        if isinstance(recommendations, PolarsDataFrame):
            self._check_duplicates_polars(recommendations)
            assert isinstance(ground_truth, PolarsDataFrame)
            return self._polars_call(recommendations, ground_truth)
        is_pandas = isinstance(recommendations, PandasDataFrame)
        recommendations = (
            self._convert_pandas_to_dict_with_score(recommendations)
            if is_pandas
            else self._convert_dict_to_dict_with_score(recommendations)
        )
        self._check_duplicates_dict(recommendations)
        ground_truth = self._convert_pandas_to_dict_without_score(ground_truth) if is_pandas else ground_truth
        assert isinstance(ground_truth, dict)
        return self._dict_call(
            list(ground_truth),
            pred_item_id=recommendations,
            ground_truth=ground_truth,
        )

    def _convert_pandas_to_dict_with_score(self, data: PandasDataFrame) -> Dict:
        return (
            data.sort_values(by=self.rating_column, ascending=False)
            .groupby(self.query_column)[self.item_column]
            .apply(list)
            .to_dict()
        )

    def _convert_dict_to_dict_with_score(self, data: Dict) -> Dict:
        converted_data = {}
        for user, items in data.items():
            is_sorted = True
            for i in range(1, len(items)):
                is_sorted &= items[i - 1][1] >= items[i][1]
                if not is_sorted:
                    break
            if not is_sorted:
                items = sorted(items, key=lambda x: x[1], reverse=True)
            converted_data[user] = [item for item, _ in items]
        return converted_data

    def _convert_pandas_to_dict_without_score(self, data: PandasDataFrame) -> Dict:
        return data.groupby(self.query_column)[self.item_column].apply(list).to_dict()

    def _dict_call(self, users: List, **kwargs: Dict) -> MetricsReturnType:
        """
        Calculating metrics in dict format.
        kwargs can contain different dicts (for example, ground_truth or train), it depends on the metric.
        """

        keys_list = sorted(kwargs.keys())
        distribution_per_user = {}
        for user in users:
            args = [kwargs[key].get(user, None) for key in keys_list]
            distribution_per_user[user] = self._get_metric_value_by_user(self.topk, *args)
        if self._mode.__name__ == "PerUser":
            return self._aggregate_results_per_user(distribution_per_user)
        distribution = np.stack(list(distribution_per_user.values()))
        assert distribution.shape[1] == len(self.topk)
        metrics = [self._mode.cpu(distribution[:, k]) for k in range(distribution.shape[1])]
        return self._aggregate_results(metrics)

    def _get_items_list_per_user_spark(
        self, recommendations: SparkDataFrame, extra_column: Optional[str] = None
    ) -> SparkDataFrame:
        recommendations = recommendations.groupby(self.query_column).agg(
            sf.sort_array(
                sf.collect_list(
                    sf.struct(*[c for c in [self.rating_column, self.item_column, extra_column] if c is not None])
                ),
                False,
            ).alias("pred")
        )
        selection = [
            self.query_column,
            sf.col(f"pred.{self.item_column}").alias("pred_item_id"),
        ]
        if extra_column:
            selection.append(sf.col(f"pred.{extra_column}").alias(extra_column))

        recommendations = recommendations.select(*selection)
        return recommendations

    def _get_items_list_per_user_polars(
        self, recommendations: PolarsDataFrame, extra_column: Optional[str] = None
    ) -> PolarsDataFrame:
        selection = [self.query_column, "pred_item_id"]
        sorting = [self.rating_column, self.item_column]
        agg = [pl.col(self.item_column)]
        if extra_column is not None:
            sorting.append(extra_column)
            agg.append(pl.col(extra_column))
            selection.append(extra_column)

        recommendations = (
            recommendations.sort(sorting, descending=True)
            .group_by(self.query_column)
            .agg(*agg)
            .rename({self.item_column: "pred_item_id"})
        )

        recommendations = recommendations.select(*selection)
        return recommendations

    def _get_items_list_per_user(
        self, recommendations: Union[SparkDataFrame, PolarsDataFrame], extra_column: Optional[str] = None
    ) -> Union[SparkDataFrame, PolarsDataFrame]:
        if isinstance(recommendations, SparkDataFrame):
            return self._get_items_list_per_user_spark(recommendations, extra_column)
        else:
            return self._get_items_list_per_user_polars(recommendations, extra_column)

    def _rearrange_columns(
        self, data: Union[SparkDataFrame, PolarsDataFrame]
    ) -> Union[SparkDataFrame, PolarsDataFrame]:
        cols = data.columns
        cols.remove(self.query_column)
        cols = [self.query_column, *sorted(cols)]
        return data.select(*cols)

    def _get_enriched_recommendations(
        self,
        recommendations: Union[PolarsDataFrame, SparkDataFrame],
        ground_truth: Union[PolarsDataFrame, SparkDataFrame],
    ) -> Union[PolarsDataFrame, SparkDataFrame]:
        if isinstance(recommendations, SparkDataFrame):
            return self._get_enriched_recommendations_spark(recommendations, ground_truth)
        else:
            return self._get_enriched_recommendations_polars(recommendations, ground_truth)

    def _get_enriched_recommendations_spark(
        self,
        recommendations: SparkDataFrame,
        ground_truth: SparkDataFrame,
    ) -> SparkDataFrame:
        true_items_by_users = ground_truth.groupby(self.query_column).agg(
            sf.collect_set(self.item_column).alias("ground_truth")
        )

        sorted_by_score_recommendations = self._get_items_list_per_user(recommendations)

        enriched_recommendations = sorted_by_score_recommendations.join(
            true_items_by_users, on=self.query_column, how="right"
        )
        return self._rearrange_columns(enriched_recommendations)

    def _get_enriched_recommendations_polars(
        self,
        recommendations: PolarsDataFrame,
        ground_truth: PolarsDataFrame,
    ) -> PolarsDataFrame:
        true_items_by_users = (
            ground_truth.group_by(self.query_column)
            .agg(pl.col(self.item_column))
            .rename({self.item_column: "ground_truth"})
        )

        sorted_by_score_recommendations = self._get_items_list_per_user(recommendations)

        enriched_recommendations = true_items_by_users.join(
            sorted_by_score_recommendations, on=self.query_column, how="left"
        )
        return self._rearrange_columns(enriched_recommendations)

    def _aggregate_results_per_user(self, distribution_per_user: Dict[Any, List[float]]) -> MetricsPerUserReturnType:
        res: MetricsPerUserReturnType = {}
        for index, val in enumerate(self.topk):
            metric_name = f"{self.__name__}@{val}"
            res[metric_name] = {}
            for user, metrics in distribution_per_user.items():
                res[metric_name][user] = metrics[index]
        return res

    def _aggregate_results(self, metrics: list) -> MetricsMeanReturnType:
        res = {}
        for index, val in enumerate(self.topk):
            metric_name = f"{self.__name__}@{val}"
            res[metric_name] = metrics[index]
        return res

    def _spark_compute(self, recs: SparkDataFrame) -> MetricsReturnType:
        """
        Calculating metrics for PySpark DataFrame.
        """
        recs_with_topk_list = recs.withColumn("k", sf.array(*[sf.lit(x) for x in self.topk]))
        distribution = self._get_metric_distribution(recs_with_topk_list)
        if self._mode.__name__ == "PerUser":
            return self._aggregate_results_per_user(distribution.rdd.collectAsMap())
        metrics = [
            self._mode.spark(distribution.select(sf.col("value").getItem(i)).withColumnRenamed(f"value[{i}]", "val"))
            for i in range(len(self.topk))
        ]
        return self._aggregate_results(metrics)

    def _polars_compute(self, recs: PolarsDataFrame) -> MetricsReturnType:
        distribution = self._get_metric_distribution(recs)
        if self._mode.__name__ == "PerUser":
            return self._aggregate_results_per_user(
                dict(
                    distribution.select(
                        self.query_column, value=pl.concat_list(pl.exclude(self.query_column))
                    ).iter_rows()
                )
            )
        metrics = [self._mode.cpu(distribution.select(column)) for column in distribution.columns[1:]]
        return self._aggregate_results(metrics)

    def _spark_call(self, recommendations: SparkDataFrame, ground_truth: SparkDataFrame) -> MetricsReturnType:
        """
        Implementation for PySpark DataFrame.
        """
        recs = self._get_enriched_recommendations(recommendations, ground_truth)
        return self._spark_compute(recs)

    def _polars_call(self, recommendations: PolarsDataFrame, ground_truth: PolarsDataFrame) -> MetricsReturnType:
        """
        Implementation for Polars DataFrame.
        """
        recs = self._get_enriched_recommendations(recommendations, ground_truth)
        return self._polars_compute(recs)

    def _get_metric_distribution(
        self, recs: Union[PolarsDataFrame, SparkDataFrame]
    ) -> Union[PolarsDataFrame, SparkDataFrame]:
        if isinstance(recs, SparkDataFrame):
            return self._get_metric_distribution_spark(recs)
        else:
            return self._get_metric_distribution_polars(recs)

    def _get_metric_distribution_spark(self, recs: SparkDataFrame) -> SparkDataFrame:
        cur_class = self.__class__
        distribution = recs.rdd.flatMap(  # pragma: no cover, due to incorrect work of coverage tool
            lambda x: [(x[0], cur_class._get_metric_value_by_user(x[-1], *x[1:-1]))]
        ).toDF(
            StructType()
            .add("user_id", recs.schema[self.query_column].dataType.typeName(), False)
            .add("value", ArrayType(DoubleType()), False)
        )
        return distribution

    def _get_metric_distribution_polars(self, recs: PolarsDataFrame) -> PolarsDataFrame:
        cur_class = self.__class__
        distribution = recs.map_rows(lambda x: (x[0], *cur_class._get_metric_value_by_user(self.topk, *x[1:])))
        distribution = distribution.rename({"column_0": self.query_column})
        distribution = distribution.rename(
            {distribution.columns[x + 1]: f"value_{self.topk[x]}" for x in range(len(self.topk))}
        )
        return distribution

    @staticmethod
    @abstractmethod
    def _get_metric_value_by_user(ks: List[int], *args: List) -> List[float]:  # pragma: no cover
        """
        Metric calculation for one user.

        :param k: depth cut-off
        :param ground_truth: test data
        :param pred: recommendations
        :return: metric value for current user
        """
        raise NotImplementedError()
