# pylint: disable=protected-access
from typing import Optional, Dict, List, Any, Tuple

from pyspark.sql import DataFrame

from replay.constants import AnyDataFrame
from replay.metrics import Metric, NDCG
from replay.models import PopRec
from replay.models.base_rec import BaseRecommender
from replay.scenarios.basescenario import BaseScenario
from replay.utils import fallback


class Fallback(BaseScenario):
    """Fill missing recommendations using fallback model.
    Behaves like a recommender and have the same interface."""

    can_predict_cold_users: bool = True

    def __init__(
        self,
        main_model: BaseRecommender,
        fallback_model: BaseRecommender = PopRec(),
        cold_model: BaseRecommender = PopRec(),
        threshold: int = 5,
    ):
        """Create recommendations with `main_model`, and fill missing with `fallback_model`.
        `relevance` of fallback_model will be decrease to keep main recommendations on top.

        :param main_model: initialized model
        :param fallback_model: initialized model
        :param cold_model: model used for cold users
        :param threshold: number of interactions by which users are divided into cold and hot
        """
        super().__init__(cold_model, threshold)
        self.main_model = main_model
        # pylint: disable=invalid-name
        self.fb_model = fallback_model

    def __str__(self):
        return f"Fallback({str(self.main_model)}, {str(self.fb_model)})"

    # pylint: disable=too-many-arguments, too-many-locals
    def _optimize(
        self,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_grid: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        criterion: Metric = NDCG(),
        k: int = 10,
        budget: int = 10,
    ) -> Tuple[Dict[str, Any]]:
        """
        Searches best parameters with optuna.

        :param train: train data
        :param test: test data
        :param user_features: user features
        :param item_features: item features
        :param param_grid: a dictionary with keys main and
            fallback containing dictionaries with search grid, where
            key is the parameter name and value is the range of possible values
            ``{param: [low, high]}``.
        :param criterion: metric to use for optimization
        :param k: recommendation list length
        :param budget: number of points to try
        :return: tuple of dictionaries with best parameters
        """
        if param_grid is None:
            param_grid = {"main": None, "fallback": None}
        self.logger.info("Optimizing main model...")
        params = self.main_model.optimize(
            train,
            test,
            user_features,
            item_features,
            param_grid["main"],
            criterion,
            k,
            budget,
        )
        self.main_model.set_params(**params)
        if self.fb_model._search_space is not None:
            self.logger.info("Optimizing fallback model...")
            fb_params = self.fb_model.optimize(
                train,
                test,
                user_features,
                item_features,
                param_grid["fallback"],
                criterion,
                k,
                budget,
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
        self.main_model.user_indexer = self.user_indexer
        self.main_model.item_indexer = self.item_indexer
        self.main_model.inv_user_indexer = self.inv_user_indexer
        self.main_model.inv_item_indexer = self.inv_item_indexer

        self.fb_model.user_indexer = self.user_indexer
        self.fb_model.item_indexer = self.item_indexer
        self.fb_model.inv_user_indexer = self.inv_user_indexer
        self.fb_model.inv_item_indexer = self.inv_item_indexer

        self.main_model._fit(log, user_features, item_features)
        self.fb_model._fit(log, user_features, item_features)

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
        extra_pred = self.fb_model._predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )
        pred = fallback(pred, extra_pred, k, id_type="idx")
        return pred
