# pylint: disable=too-many-arguments
# pragma: no cover
from abc import abstractmethod
from typing import Optional, Union, Iterable, Dict, List, Any, Tuple

from pyspark.sql import DataFrame

from replay.data import AnyDataFrame
from replay.preprocessing.filters import filter_by_min_count
from replay.metrics import Metric, NDCG
from replay.models.base_rec import BaseRecommender
from replay.utils.spark_utils import convert2spark, get_unique_entities


class BaseScenario(BaseRecommender):
    """Base scenario class"""

    can_predict_cold_users: bool = False

    def __init__(self, cold_model, threshold=5):
        self.threshold = threshold
        self.cold_model = cold_model
        self.hot_users = None

    # TO DO: add save/load for scenarios
    @property
    def _init_args(self):
        return {"threshold": self.threshold}

    def fit(
        self,
        log: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
    ) -> None:
        """
        :param log: input DataFrame ``[user_id, item_id, timestamp, relevance]``
        :param user_features: user features ``[user_id, timestamp]`` + feature columns
        :param item_features: item features ``[item_id, timestamp]`` + feature columns
        :return:
        """
        hot_data = filter_by_min_count(log, self.threshold, "user_idx")
        self.hot_users = hot_data.select("user_idx").distinct()
        self._fit_wrap(hot_data, user_features, item_features)
        self.cold_model._fit_wrap(log, user_features, item_features)

    # pylint: disable=too-many-arguments
    def predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Get recommendations

        :param log: historical log of interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param k: length of recommendation lists, should be less that the total number of ``items``
        :param users: users to create recommendations for
            dataframe containing ``[user_id]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_id]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be``0``.
        :param user_features: user features
            ``[user_id , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_id , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :return: recommendation dataframe
            ``[user_id, item_id, relevance]``
        """
        log = convert2spark(log)
        users = users or log or user_features or self.fit_users
        users = get_unique_entities(users, "user_idx")
        hot_data = filter_by_min_count(log, self.threshold, "user_idx")
        hot_users = hot_data.select("user_idx").distinct()
        if not self.can_predict_cold_users:
            hot_users = hot_users.join(self.hot_users)
        hot_users = hot_users.join(users, on="user_idx", how="inner")

        hot_pred = self._predict_wrap(
            log=hot_data,
            k=k,
            users=hot_users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
        )
        if log is not None:
            cold_data = log.join(self.hot_users, how="anti", on="user_idx")
        else:
            cold_data = None
        cold_users = users.join(self.hot_users, how="anti", on="user_idx")
        cold_pred = self.cold_model._predict_wrap(
            log=cold_data,
            k=k,
            users=cold_users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
        )
        return hot_pred.union(cold_pred)

    def fit_predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Train and get recommendations

        :param log: historical log of interactions
            ``[user_id, item_id, timestamp, relevance]``
        :param k: length of recommendation lists, should be less that the total number of ``items``
        :param users: users to create recommendations for
            dataframe containing ``[user_id]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_id]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be``0``.
        :param user_features: user features
            ``[user_id , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_id , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :return: recommendation dataframe
            ``[user_id, item_id, relevance]``
        """
        self.fit(log, user_features, item_features)
        return self.predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )

    # pylint: disable=too-many-arguments, too-many-locals
    def optimize(
        self,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_borders: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        criterion: Metric = NDCG(),
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ) -> Tuple[Dict[str, Any]]:
        """
        Searches best parameters with optuna.

        :param train: train data
        :param test: test data
        :param user_features: user features
        :param item_features: item features
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
            train,
            test,
            user_features,
            item_features,
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
                train,
                test,
                user_features,
                item_features,
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
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_borders: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        criterion: Metric = NDCG(),
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ):
        pass
