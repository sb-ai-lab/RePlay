# pylint: disable=protected-access
from typing import Optional, Dict, List, Any, Tuple, Union, Iterable

from pyspark.sql import DataFrame

from replay.data import AnyDataFrame
from replay.preprocessing.filters import filter_by_min_count
from replay.metrics import Metric, NDCG
from replay.models import PopRec
from replay.models.base_rec import BaseRecommender
from replay.utils.spark_utils import fallback, get_unique_entities


class Fallback(BaseRecommender):
    """Fill missing recommendations using fallback model.
    Behaves like a recommender and have the same interface."""

    can_predict_cold_users: bool = True

    def __init__(
        self,
        main_model: BaseRecommender,
        fallback_model: BaseRecommender = PopRec(),
        threshold: int = 0,
    ):
        """Create recommendations with `main_model`, and fill missing with `fallback_model`.
        `relevance` of fallback_model will be decrease to keep main recommendations on top.

        :param main_model: initialized model
        :param fallback_model: initialized model
        :param threshold: number of interactions by which users are divided into cold and hot
        """
        self.threshold = threshold
        self.hot_users = None
        self.main_model = main_model
        # pylint: disable=invalid-name
        self.fb_model = fallback_model

    # TO DO: add save/load for scenarios
    @property
    def _init_args(self):
        return {"threshold": self.threshold}

    def __str__(self):
        return f"Fallback_{str(self.main_model)}_{str(self.fb_model)}"

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
        self.fb_model._fit_wrap(log, user_features, item_features)

    # pylint: disable=too-many-arguments
    def predict(
        self,
        log: DataFrame,
        k: int,
        users: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Get recommendations

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: length of recommendation lists, should be less that the total number of ``items``
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :return: recommendation dataframe
            ``[user_idx, item_idx, relevance]``
        """
        users = users or log or user_features or self.fit_users
        users = get_unique_entities(users, "user_idx")
        hot_data = filter_by_min_count(log, self.threshold, "user_idx")
        hot_users = hot_data.select("user_idx").distinct()
        hot_users = hot_users.join(self.hot_users, on="user_idx")
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
        cold_pred = self.fb_model._predict_wrap(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
        )
        pred = fallback(hot_pred, cold_pred, k)
        return pred

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
        :param param_borders: a dictionary with keys main and
            fallback containing dictionaries with search grid, where
            key is the parameter name and value is the range of possible values
            ``{param: [low, high]}``.
        :param criterion: metric to use for optimization
        :param k: recommendation list length
        :param budget: number of points to try
        :param new_study: keep searching with previous study or start a new study
        :return: tuple of dictionaries with best parameters
        """
        if param_borders is None:
            param_borders = {"main": None, "fallback": None}
        self.logger.info("Optimizing main model...")
        params = self.main_model.optimize(
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
        self.main_model.set_params(**params)
        if self.fb_model._search_space is not None:
            self.logger.info("Optimizing fallback model...")
            fb_params = self.fb_model.optimize(
                train,
                test,
                user_features,
                item_features,
                param_borders["fallback"],
                criterion,
                k,
                budget,
                new_study,
            )
            self.fb_model.set_params(**fb_params)
        else:
            fb_params = None
        return params, fb_params

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.main_model._fit_wrap(log, user_features, item_features)
        self.fb_model._fit_wrap(log, user_features, item_features)

    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        pred = self.main_model._predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )
        return pred
