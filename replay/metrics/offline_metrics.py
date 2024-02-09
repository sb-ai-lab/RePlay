import warnings
from typing import Dict, List, Optional, Tuple, Union

from replay.utils import PandasDataFrame, SparkDataFrame

from .base_metric import Metric, MetricsDataFrameLike, MetricsReturnType
from .coverage import Coverage
from .novelty import Novelty
from .recall import Recall
from .surprisal import Surprisal


# pylint: disable=too-few-public-methods
class OfflineMetrics:
    """
    Designed for efficient calculation of offline metrics provided by the RePlay.
    If you need to calculate multiple metrics for the same input data,
    then using this class is much more efficient than calculating metrics individually.

    For example, you want to calculate several metrics with different parameters.
    When calling offline metrics with the specified metrics,
    the common part of these metrics will be computed only once.

    >>> from replay.metrics import *
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
    >>> groundtruth
       query_id  item_id
    0         1        5
    1         1        6
    2         1        7
    3         1        8
    4         1        9
    5         1       10
    6         2        6
    7         2        7
    8         2        4
    9         2       10
    10        2       11
    11        3        1
    12        3        2
    13        3        3
    14        3        4
    15        3        5
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
    >>> base_rec
       query_id  item_id  rating
    0        1        3    0.5
    1        1        7    0.5
    2        1        2    0.7
    3        2        5    0.6
    4        2        8    0.6
    5        2        3    0.3
    6        3        4    1.0
    7        3        9    0.5
    >>> from replay.metrics import Median, ConfidenceInterval, PerUser
    >>> metrics = [
    ...     Precision(2),
    ...     Precision(2, mode=PerUser()),
    ...     Precision(2, mode=Median()),
    ...     Precision(2, mode=ConfidenceInterval(alpha=0.95)),
    ...     Recall(2),
    ...     MAP(2),
    ...     MRR(2),
    ...     NDCG(2),
    ...     HitRate(2),
    ...     RocAuc(2),
    ...     Coverage(2),
    ...     Novelty(2),
    ...     Surprisal(2),
    ... ]
    >>> OfflineMetrics(metrics)(recommendations, groundtruth, train)
    {'Precision@2': 0.3333333333333333,
     'Precision-PerUser@2': {1: 0.5, 2: 0.0, 3: 0.5},
     'Precision-Median@2': 0.5,
     'Precision-ConfidenceInterval@2': 0.32666066409000905,
     'Recall@2': 0.12222222222222223,
     'MAP@2': 0.25,
     'MRR@2': 0.5,
     'NDCG@2': 0.3333333333333333,
     'HitRate@2': 0.6666666666666666,
     'RocAuc@2': 0.3333333333333333,
     'Coverage@2': 0.5555555555555556,
     'Novelty@2': 0.3333333333333333,
     'Surprisal@2': 0.6845351232142715}
    >>> metrics = [
    ...     Precision(2),
    ...     Unexpectedness([1, 2]),
    ...     Unexpectedness([1, 2], mode=PerUser()),
    ... ]
    >>> OfflineMetrics(metrics)(
    ...     recommendations,
    ...     groundtruth,
    ...     train,
    ...     base_recommendations={"ALS": base_rec, "KNN": recommendations}
    ... )
    {'Precision@2': 0.3333333333333333,
     'Unexpectedness_ALS@1': 0.3333333333333333,
     'Unexpectedness_ALS@2': 0.16666666666666666,
     'Unexpectedness_KNN@1': 0.0,
     'Unexpectedness_KNN@2': 0.0,
     'Unexpectedness-PerUser_ALS@1': {1: 1.0, 2: 0.0, 3: 0.0},
     'Unexpectedness-PerUser_ALS@2': {1: 0.5, 2: 0.0, 3: 0.0},
     'Unexpectedness-PerUser_KNN@1': {1: 0.0, 2: 0.0, 3: 0.0},
     'Unexpectedness-PerUser_KNN@2': {1: 0.0, 2: 0.0, 3: 0.0}}
    <BLANKLINE>
    """

    _metrics_call_requirement_map: Dict[str, List[str]] = {
        "HitRate": ["ground_truth"],
        "MAP": ["ground_truth"],
        "NDCG": ["ground_truth"],
        "RocAuc": ["ground_truth"],
        "Coverage": ["train"],
        "Novelty": ["train"],
        "Surprisal": ["train"],
        "MRR": ["ground_truth"],
        "Precision": ["ground_truth"],
        "Recall": ["ground_truth"],
    }

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        metrics: List[Metric],
        query_column: str = "query_id",
        item_column: str = "item_id",
        rating_column: str = "rating",
        category_column: str = "category_id",
        allow_caching: bool = True,
    ):
        """
        :param metrics: (list of metrics): List of metrics to be calculated.
        :param user_column: (str): The name of the user column.
            Note that you do not need to specify the value of this parameter for each metric separately.
            It is enough to specify the value of this parameter here once.
        :param item_column: (str): The name of the item column.
            Note that you do not need to specify the value of this parameter for each metric separately.
            It is enough to specify the value of this parameter here once.
        :param score_column: (str): The name of the score column.
            Note that you do not need to specify the value of this parameter for each metric separately.
            It is enough to specify the value of this parameter here once.
        :param category_column: (str): The name of the category column.
            Note that you do not need to specify the value of this parameter for each metric separately.
            It is enough to specify the value of this parameter here once.

            It is used only for calculating the ``Diversity`` metric.
            If you don't calculate this metric, you can omit this parameter.
        :param allow_caching: (bool): The flag for using caching to optimize calculations.
            Default: ``True``.
        """
        self.unexpectedness_metric: List[Metric] = []
        self.diversity_metric: List[Metric] = []
        self.main_metrics: List[Metric] = []
        self._allow_caching = allow_caching

        for metric in metrics:
            metric.query_column = query_column
            metric.item_column = item_column
            metric.rating_column = rating_column
            if metric.__class__.__name__ in ["Unexpectedness"]:
                self.unexpectedness_metric.append(metric)
            elif metric.__class__.__name__ in ["CategoricalDiversity"]:
                metric.category_column = category_column
                self.diversity_metric.append(metric)
            else:
                self.main_metrics.append(metric)

        self.metrics = self.main_metrics

    def _get_enriched_recommendations(
        self,
        recommendations: SparkDataFrame,
        ground_truth: SparkDataFrame,
        train: Optional[SparkDataFrame],
    ) -> Tuple[Dict[str, SparkDataFrame], Optional[SparkDataFrame]]:
        if len(self.main_metrics) == 0:
            return {}, train
        result_dict = {}
        query_column = self.main_metrics[0].query_column
        item_column = self.main_metrics[0].item_column
        rating_column = self.main_metrics[0].rating_column
        default_metric = Recall(
            topk=2,
            query_column=query_column,
            item_column=item_column,
            rating_column=rating_column,
        )
        default_metric._check_duplicates_spark(recommendations)
        unchanged_recs = recommendations

        # pylint: disable=too-many-function-args
        result_dict["default"] = default_metric._get_enriched_recommendations(
            recommendations, ground_truth
        )

        for metric in self.metrics:
            # find Coverage
            if metric.__class__.__name__ == "Coverage":
                # pylint: disable=protected-access
                result_dict["Coverage"] = Coverage(
                    topk=2,
                    query_column=query_column,
                    item_column=item_column,
                    rating_column=rating_column,
                )._get_enriched_recommendations(recommendations)

            # find Novelty
            if metric.__class__.__name__ == "Novelty" and train is not None:
                novelty_metric = Novelty(
                    topk=2,
                    query_column=query_column,
                    item_column=item_column,
                    rating_column=rating_column,
                )
                cur_recs = novelty_metric._get_enriched_recommendations(
                    unchanged_recs, train
                ).withColumnRenamed("ground_truth", "train")
                cur_recs = metric._rearrange_columns(cur_recs)
                result_dict["Novelty"] = cur_recs

            # find Surprisal
            if metric.__class__.__name__ == "Surprisal" and train is not None:
                result_dict["Surprisal"] = Surprisal(
                    topk=2,
                    query_column=query_column,
                    item_column=item_column,
                    rating_column=rating_column,
                )._get_enriched_recommendations(unchanged_recs, train)

        return result_dict, train

    # pylint: disable=no-self-use
    def _cache_dataframes(self, dataframes: Dict[str, SparkDataFrame]) -> None:
        for data in dataframes.values():
            data.cache()

    # pylint: disable=no-self-use
    def _unpersist_dataframes(self, dataframes: Dict[str, SparkDataFrame]) -> None:
        for data in dataframes.values():
            data.unpersist()

    def _calculate_metrics(
        self,
        enriched_recs_dict: Dict[str, SparkDataFrame],
        train: Optional[SparkDataFrame] = None,
    ) -> MetricsReturnType:
        result: Dict = {}
        for metric in self.metrics:
            metric_args = {}
            if metric.__class__.__name__ == "Coverage" and train is not None:
                metric_args["recs"] = enriched_recs_dict["Coverage"]
                metric_args["train"] = train
            elif metric.__class__.__name__ == "Surprisal":
                metric_args["recs"] = enriched_recs_dict["Surprisal"]
            elif metric.__class__.__name__ == "Novelty":
                metric_args["recs"] = enriched_recs_dict["Novelty"]
            else:
                metric_args["recs"] = enriched_recs_dict["default"]

            # pylint: disable=protected-access
            result.update(metric._spark_compute(**metric_args))
        return result

    # pylint: disable=no-self-use
    def _check_dataframes_types(
        self,
        recommendations: MetricsDataFrameLike,
        ground_truth: MetricsDataFrameLike,
        train: Optional[MetricsDataFrameLike],
        base_recommendations: Optional[
            Union[MetricsDataFrameLike, Dict[str, MetricsDataFrameLike]]
        ],
    ) -> None:
        types = set()
        types.add(type(recommendations))
        types.add(type(ground_truth))
        if train is not None:
            types.add(type(train))
        if isinstance(base_recommendations, dict):
            for _, df in base_recommendations.items():
                if not isinstance(df, list):
                    types.add(type(df))
                else:
                    types.add(dict)
                    break
        elif base_recommendations is not None:
            types.add(type(base_recommendations))

        if len(types) != 1:
            raise ValueError("All given data frames must have the same type")

    def _check_query_column_present(
        self,
        dataset: MetricsDataFrameLike,
        query_column: str,
        dataset_name: str,
    ):
        """
        Checks that query column presented in provided dataframe.

        :param dataset: input dataframe.
        :param query_column: name of query column.
        :param dataset_name: name of dataframe.

        :raises KeyError: if query column not found in dataframe.
        """
        if isinstance(dataset, SparkDataFrame):
            dataset_names = dataset.schema.names
        elif isinstance(dataset, PandasDataFrame):
            dataset_names = dataset.columns

        if not isinstance(dataset, dict) and query_column not in dataset_names:
            raise KeyError(f"Query column {query_column} is not present in {dataset_name} dataframe")

    def _get_unique_queries(
        self,
        dataset: MetricsDataFrameLike,
        query_column: str,
    ):
        """
        Returns unique queries from provided dataframe.

        :param dataset: input dataframe.
        :param query_column: name of query column.

        :returns: set of unique queries.
        """
        if isinstance(dataset, SparkDataFrame):
            return set(dataset.select(query_column).distinct().toPandas()[query_column])
        elif isinstance(dataset, PandasDataFrame):
            return set(dataset[query_column].unique())
        else:
            return set(dataset.keys())

    def _check_contains(self, queries: set, other_queries: set, dataset_name: str):
        """
        Checks all queries from the first set are presented in the second one.
        Throws warning otherwise.

        :param queries: first set of queries.
        :param other_queries: second set of queries.
        :param dataset_name: name of dataset to specify in warning message.
        """
        if queries.issubset(other_queries) is False:
            warnings.warn(f"{dataset_name} contains queries that are not presented in recommendations")

    def __call__(  # pylint: disable=too-many-branches, too-many-locals
        self,
        recommendations: MetricsDataFrameLike,
        ground_truth: MetricsDataFrameLike,
        train: Optional[MetricsDataFrameLike] = None,
        base_recommendations: Optional[
            Union[MetricsDataFrameLike, Dict[str, MetricsDataFrameLike]]
        ] = None,
    ) -> Dict[str, float]:
        """
        Compute metrics.

        :param recommendations: (PySpark DataFrame or Pandas DataFrame or dict): model predictions.
            If DataFrame then it must contains user, item and score columns.
            If dict then key represents user_ids, value represents list of tuple(item_id, score).
        :param ground_truth: (PySpark DataFrame or Pandas DataFrame or dict): test data.
            If DataFrame then it must contains user and item columns.
            If dict then key represents user_ids, value represents list of item_ids.
        :param train: (PySpark DataFrame or Pandas DataFrame or dict, optional): train data.
            If DataFrame then it must contains user and item columns.
            If dict then key represents user_ids, value represents list of item_ids.
            Default: ``None``.
        :param base_recommendations: (PySpark DataFrame or Pandas DataFrame or dict or Dict[str, DataFrameLike]):
            predictions from baseline model.
            If DataFrame then it must contains user, item and score columns.
            If dict then key represents user_ids, value represents list of tuple(item_id, score).
            If ``Unexpectedness`` is not in given metrics list, then you can omit this parameter.
            If it is necessary to calculate the value of metrics on several dataframes,
            then you need to submit a dict(key - name of the data frame, value - DataFrameLike).
            For a better understanding, check out the examples.
            Default: ``None``.

        :return: metric values
        """
        self._check_dataframes_types(
            recommendations, ground_truth, train, base_recommendations
        )

        if len(self.main_metrics) > 0:
            query_column = self.main_metrics[0].query_column
        elif len(self.unexpectedness_metric) > 0:
            query_column = self.unexpectedness_metric[0].query_column
        else:
            query_column = self.diversity_metric[0].query_column

        self._check_query_column_present(recommendations, query_column, "recommendations")
        recs_queries = self._get_unique_queries(recommendations, query_column)

        self._check_query_column_present(ground_truth, query_column, "ground_truth")
        self._check_contains(recs_queries, self._get_unique_queries(ground_truth, query_column), "ground_truth")

        if train is not None:
            self._check_query_column_present(train, query_column, "train")
            self._check_contains(
                recs_queries,
                self._get_unique_queries(train, query_column),
                "train"
            )
        if base_recommendations is not None:
            if (not isinstance(base_recommendations, dict)
                    or isinstance(next(iter(base_recommendations.values())), list)):
                base_recommendations = {"base_recommendations": base_recommendations}
            for name, dataset in base_recommendations.items():
                self._check_query_column_present(dataset, query_column, name)
                self._check_contains(
                    recs_queries,
                    self._get_unique_queries(dataset, query_column),
                    name
                )

        result = {}
        if isinstance(recommendations, SparkDataFrame):
            assert isinstance(ground_truth, SparkDataFrame)
            assert train is None or isinstance(train, SparkDataFrame)
            enriched_recs_dict, train = self._get_enriched_recommendations(
                recommendations, ground_truth, train
            )

            if self._allow_caching:
                self._cache_dataframes(enriched_recs_dict)
            result.update(self._calculate_metrics(enriched_recs_dict, train))
            if self._allow_caching:
                self._unpersist_dataframes(enriched_recs_dict)
        else:  # Calculating metrics in dict format
            current_map: Dict[str, Union[PandasDataFrame, Dict]] = {
                "ground_truth": ground_truth,
                "train": train,
            }
            for metric in self.metrics:
                args_to_call: Dict[str, Union[PandasDataFrame, Dict]] = {
                    "recommendations": recommendations
                }
                for data_name in self._metrics_call_requirement_map[
                    str(metric.__class__.__name__)
                ]:
                    args_to_call[data_name] = current_map[data_name]
                result.update(metric(**args_to_call))
        unexpectedness_result = {}
        diversity_result = {}

        if len(self.unexpectedness_metric) != 0:
            if base_recommendations is None:
                raise ValueError(
                    "Can not calculate Unexpectedness because base_recommendations is None"
                )
            if isinstance(base_recommendations, dict) and not isinstance(
                list(base_recommendations.values())[0], list
            ):
                for unexp in self.unexpectedness_metric:
                    for model_name in base_recommendations:
                        cur_result = unexp(
                            recommendations, base_recommendations[model_name]
                        )
                        for metric_name in cur_result:
                            splitted = metric_name.split("@")
                            splitted[0] += "_" + model_name
                            unexpectedness_result["@".join(splitted)] = cur_result[
                                metric_name
                            ]

        if len(self.diversity_metric) != 0:
            for diversity in self.diversity_metric:
                diversity_result.update(diversity(recommendations))

        return {**result, **unexpectedness_result, **diversity_result}
