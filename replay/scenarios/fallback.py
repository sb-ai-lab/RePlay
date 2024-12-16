from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from replay.data import Dataset
from replay.metrics import NDCG, Metric
from replay.models import PopRec
from replay.models.base_rec import BaseRecommender
from replay.preprocessing.filters import MinCountFilter
from replay.utils import SparkDataFrame
from replay.utils.spark_utils import fallback, get_unique_entities


class Fallback(BaseRecommender):
    """Fill missing recommendations using fallback model.
    Behaves like a recommender and have the same interface."""

    can_predict_cold_queries: bool = True

    def __init__(
        self,
        main_model: BaseRecommender,
        fallback_model: BaseRecommender = PopRec(),
        threshold: int = 0,
    ):
        """Create recommendations with `main_model`, and fill missing with `fallback_model`.
        `rating` of fallback_model will be decrease to keep main recommendations on top.

        :param main_model: initialized model
        :param fallback_model: initialized model
        :param threshold: number of interactions by which queries are divided into cold and hot
        """
        self.threshold = threshold
        self.hot_queries = None
        self.main_model = main_model
        self.fb_model = fallback_model

    # TODO: add save/load for scenarios
    @property
    def _init_args(self):
        return {"threshold": self.threshold}

    def __str__(self):
        return f"Fallback_{self.main_model!s}_{self.fb_model!s}"

    def fit(
        self,
        dataset: Dataset,
    ) -> None:
        """
        :param dataset: input Dataset with interactions and features ``[user_id, item_id, timestamp, rating]``
        :return:
        """
        query_column = dataset.feature_schema.query_id_column

        hot_data = MinCountFilter(self.threshold, query_column).transform(dataset.interactions)
        self.hot_queries = hot_data.select(query_column).distinct()
        hot_dataset = Dataset(
            feature_schema=dataset.feature_schema,
            interactions=hot_data,
            query_features=dataset.query_features,
            item_features=dataset.item_features,
            check_consistency=True,
            categorical_encoded=False,
        )
        self._fit_wrap(hot_dataset)
        self.fb_model._fit_wrap(dataset)

    def predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[SparkDataFrame, Iterable]] = None,
        items: Optional[Union[SparkDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        """
        Get recommendations

        :param dataset: historical interactions
            ``[user_idx, item_idx, timestamp, rating]``
        :param k: length of recommendation lists, should be less that the total number of ``items``
        :param queries: queries to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all queries from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``rating`` for them will be``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :return: recommendation dataframe
            ``[user_idx, item_idx, rating]``
        """
        queries = queries or dataset.interactions or dataset.query_features or self.fit_queries
        queries = get_unique_entities(queries, self.query_column)
        hot_data = MinCountFilter(self.threshold, self.query_column).transform(dataset.interactions)
        hot_queries = hot_data.select(self.query_column).distinct()
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
        cold_pred = self.fb_model._predict_wrap(
            dataset=dataset,
            k=k,
            queries=queries,
            items=items,
            filter_seen_items=filter_seen_items,
        )
        pred = fallback(hot_pred, cold_pred, k)
        return pred

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
            train_dataset,
            test_dataset,
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
                train_dataset,
                test_dataset,
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
        dataset: Dataset,
    ) -> None:
        self.main_model._fit_wrap(dataset)
        self.fb_model._fit_wrap(dataset)

    def _predict(
        self,
        dataset: Dataset,
        k: int,
        queries: SparkDataFrame,
        items: SparkDataFrame,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        pred = self.main_model._predict(
            dataset,
            k,
            queries,
            items,
            filter_seen_items,
        )
        return pred
