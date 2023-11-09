# pylint: disable=too-many-arguments
# pragma: no cover
from abc import abstractmethod
from typing import Optional, Union, Iterable, Dict, List, Any, Tuple

from pyspark.sql import DataFrame

from replay.data import AnyDataFrame, Dataset
from replay.preprocessing.filters import filter_by_min_count
from replay.metrics import Metric, NDCG
from replay.models.base_rec import BaseRecommender
from replay.utils.spark_utils import get_unique_entities


class BaseScenario(BaseRecommender):
    """Base scenario class"""

    can_predict_cold_queries: bool = False

    def __init__(self, cold_model, threshold=5):    # pragma: no cover
        self.threshold = threshold
        self.cold_model = cold_model
        self.hot_queries = None

    # TO DO: add save/load for scenarios
    @property
    def _init_args(self):
        return {"threshold": self.threshold}

    def fit(
        self,
        dataset: Dataset,
    ) -> None:
        """
        :param dataset: input Dataset with interactions and features ``[user_id, item_id, timestamp, rating]``
        :return:
        """
        self.query_column = dataset.feature_schema.query_id_column
        self.item_column = dataset.feature_schema.item_id_column
        self.rating_column = dataset.feature_schema.interactions_rating_column
        self.timestamp_column = dataset.feature_schema.interactions_timestamp_column

        hot_data = filter_by_min_count(dataset.interactions, self.threshold, self.query_column)
        self.hot_queries = hot_data.select(self.query_column).distinct()
        hot_dataset = Dataset(
            feature_schema=dataset.feature_schema,
            interactions=hot_data,
            query_features=dataset.query_features,
            item_features=dataset.item_features,
            check_consistency=True,
            categorical_encoded=False,
        )
        self._fit_wrap(hot_dataset)
        self.cold_model._fit_wrap(dataset)

    # pylint: disable=too-many-arguments
    def predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Get recommendations

        :param dataset: historical interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param k: length of recommendation lists, should be less that the total number of ``items``
        :param queries: queries to create recommendations for
            dataframe containing ``[user_id]`` or ``array-like``;
            if ``None``, recommend to all queries from ``interactions``
        :param items: candidate items for recommendations
            dataframe containing ``[item_id]`` or ``array-like``;
            if ``None``, take all items from ``interactions``.
            If it contains new items, ``relevance`` for them will be``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``interactions``.
        :return: recommendation dataframe
            ``[user_id, item_id, relevance]``
        """
        queries = queries or dataset.interactions or dataset.query_features or self.fit_queries
        queries = get_unique_entities(queries, self.query_column)
        hot_data = filter_by_min_count(dataset.interactions, self.threshold, self.query_column)
        hot_queries = hot_data.select(self.query_column).distinct()
        if not self.can_predict_cold_queries:
            hot_queries = hot_queries.join(self.hot_queries)
        hot_queries = hot_queries.join(queries, on=self.query_column, how="inner")

        hot_dataset = Dataset(
            feature_schema=dataset.feature_schema,
            interactions=hot_data,
            query_features=dataset.query_features,
            item_features=dataset.item_features,
            check_consistency=True,
            categorical_encoded=False,
        )

        hot_pred = self._predict_wrap(
            dataset=hot_dataset,
            k=k,
            queries=hot_queries,
            items=items,
            filter_seen_items=filter_seen_items,
        )
        if dataset is not None:
            cold_data = dataset.interactions.join(self.hot_queries, how="anti", on=self.query_column)
        else:
            cold_data = None
        cold_queries = queries.join(self.hot_queries, how="anti", on=self.query_column)

        cold_dataset = Dataset(
            feature_schema=dataset.feature_schema,
            interactions=cold_data,
            query_features=dataset.query_features,
            item_features=dataset.item_features,
            check_consistency=True,
            categorical_encoded=False,
        )

        cold_pred = self.cold_model._predict_wrap(
            dataset=cold_dataset,
            k=k,
            queries=cold_queries,
            items=items,
            filter_seen_items=filter_seen_items,
        )
        return hot_pred.union(cold_pred)

    def fit_predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Train and get recommendations

        :param dataset: historical interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param k: length of recommendation lists, should be less that the total number of ``items``
        :param queries: queries to create recommendations for
            dataframe containing ``[user_id]`` or ``array-like``;
            if ``None``, recommend to all queries from ``interactions``
        :param items: candidate items for recommendations
            dataframe containing ``[item_id]`` or ``array-like``;
            if ``None``, take all items from ``interactions``.
            If it contains new items, ``relevance`` for them will be``0``.
        :param user_features: user features
            ``[user_id , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_id , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``interactions``.
        :return: recommendation dataframe
            ``[user_id, item_id, relevance]``
        """
        self.fit(dataset)
        return self.predict(
            dataset,
            k,
            queries,
            items,
            filter_seen_items,
        )

    # pylint: disable=too-many-arguments, too-many-locals
    def optimize(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        param_borders: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        criterion: Metric = NDCG,
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ) -> Tuple[Dict[str, Any]]:
        """
        Searches best parameters with optuna.

        :param train_dataset: train data
        :param test_dataset: test data
        :param param_borders: a dictionary with search grid, where
            key is the parameter name and value is
            the range of possible values``{param: [low, high]}``.
        :param criterion: metric to use for optimization
        :param k: recommendation list length
        :param budget: number of points to try
        :param new_study: keep searching with previous study or start a new study
        :return: dictionary with best parameters
        """
        if param_borders is None:
            param_borders = {"main": None, "cold": None}
        self.logger.info("Optimizing main model...")
        params = self._optimize(
            train_dataset,
            test_dataset,
            param_borders["main"],
            criterion,
            k,
            budget,
            new_study,
        )
        if not isinstance(params, tuple):
            self.set_params(**params)
        if self.cold_model._search_space is not None:
            self.logger.info("Optimizing cold model...")
            cold_params = self.cold_model._optimize(
                train_dataset,
                test_dataset,
                param_borders["cold"],
                criterion,
                k,
                budget,
                new_study,
            )
            if not isinstance(cold_params, tuple):
                self.cold_model.set_params(**cold_params)
        else:
            cold_params = None
        return params, cold_params

    @abstractmethod
    def _optimize(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        param_borders: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        criterion: Metric = NDCG,
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ):  # pragma: no cover
        pass
