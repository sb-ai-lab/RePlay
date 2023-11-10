# pylint: disable=too-many-lines
"""
Base abstract classes:
- BaseRecommender - the simplest base class
- Recommender - base class for models that fit on interactions
- HybridRecommender - base class for models that accept query or item features
- QueryRecommender - base class that accepts only query features, but not item features
- NeighbourRec - base class that requires interactions at prediction time
- ItemVectorModel - class for models which provides items' vectors.
    Implements similar items search.
- NonPersonalizedRecommender - base class for non-personalized recommenders
    with popularity statistics
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from os.path import join
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import pandas as pd
from numpy.random import default_rng
from optuna import create_study
from optuna.samplers import TPESampler
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf

from replay.data import get_rec_schema
from replay.metrics import Metric, NDCG
from replay.optimization.optuna_objective import SplitData, MainObjective
from replay.utils.session_handler import State
from replay.utils.spark_utils import (
    cache_temp_view,
    convert2spark,
    cosine_similarity,
    drop_temp_view,
    filter_cold,
    get_unique_entities,
    get_top_k,
    get_top_k_recs,
    return_recs,
    vector_euclidean_distance_similarity,
    vector_dot,
    save_picklable_to_parquet,
    load_pickled_from_parquet,
)
from replay.data import Dataset


# pylint: disable=too-few-public-methods
class IsSavable(ABC):
    """
    Common methods and attributes for saving and loading RePlay models
    """

    @property
    @abstractmethod
    def _init_args(self):
        """
        Dictionary of the model attributes passed during model initialization.
        Used for model saving and loading
        """

    @property
    def _dataframes(self):
        """
        Dictionary of the model dataframes required for inference.
        Used for model saving and loading
        """
        return {}

    def _save_model(self, path: str):
        pass

    def _load_model(self, path: str):
        pass


class RecommenderCommons:
    """
    Common methods and attributes of RePlay models for caching, setting parameters and logging
    """

    _logger: Optional[logging.Logger] = None
    cached_dfs: Optional[Set] = None
    query_column: str
    item_column: str
    rating_column: str
    timestamp_column: str

    def set_params(self, **params: Dict[str, Any]) -> None:
        """
        Set model parameters

        :param params: dictionary param name - param value
        :return:
        """
        for param, value in params.items():
            setattr(self, param, value)
        self._clear_cache()

    def _clear_cache(self):
        """
        Clear spark cache
        """

    def __str__(self):
        return type(self).__name__

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    def _cache_model_temp_view(self, df: DataFrame, df_name: str) -> None:
        """
        Create Spark SQL temporary view for df, cache it and add temp view name to self.cached_dfs.
        Temp view name is : "id_<python object id>_model_<RePlay model name>_<df_name>"
        """
        full_name = f"id_{id(self)}_model_{str(self)}_{df_name}"
        cache_temp_view(df, full_name)

        if self.cached_dfs is None:
            self.cached_dfs = set()
        self.cached_dfs.add(full_name)

    def _clear_model_temp_view(self, df_name: str) -> None:
        """
        Uncache and drop Spark SQL temporary view and remove from self.cached_dfs
        Temp view to replace will be constructed as
        "id_<python object id>_model_<RePlay model name>_<df_name>"
        """
        full_name = f"id_{id(self)}_model_{str(self)}_{df_name}"
        drop_temp_view(full_name)
        if self.cached_dfs is not None:
            self.cached_dfs.discard(full_name)


# pylint: disable=too-many-instance-attributes
class BaseRecommender(RecommenderCommons, IsSavable, ABC):
    """Base recommender"""

    model: Any
    can_predict_cold_queries: bool = False
    can_predict_cold_items: bool = False
    _search_space: Optional[
        Dict[str, Union[str, Sequence[Union[str, int, float]]]]
    ] = None
    _objective = MainObjective
    study = None
    criterion = None
    fit_queries: DataFrame
    fit_items: DataFrame
    _num_queries: int
    _num_items: int
    _query_dim_size: int
    _item_dim_size: int

    # pylint: disable=too-many-arguments, too-many-locals, no-member
    def optimize(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        param_borders: Optional[Dict[str, List[Any]]] = None,
        criterion: Metric = NDCG,
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Searches the best parameters with optuna.

        :param train_dataset: train data
        :param test_dataset: test data
        :param param_borders: a dictionary with search borders, where
            key is the parameter name and value is the range of possible values
            ``{param: [low, high]}``. In case of categorical parameters it is
            all possible values: ``{cat_param: [cat_1, cat_2, cat_3]}``.
        :param criterion: metric to use for optimization
        :param k: recommendation list length
        :param budget: number of points to try
        :param new_study: keep searching with previous study or start a new study
        :return: dictionary with best parameters
        """
        self.query_column = train_dataset.feature_schema.query_id_column
        self.item_column = train_dataset.feature_schema.item_id_column
        self.rating_column = train_dataset.feature_schema.interactions_rating_column
        self.timestamp_column = train_dataset.feature_schema.interactions_timestamp_column

        self.criterion = criterion(
            query_column=self.query_column,
            item_column=self.item_column,
            rating_column=self.rating_column,
        )

        if self._search_space is None:
            self.logger.warning(
                "%s has no hyper parameters to optimize", str(self)
            )
            return None

        if self.study is None or new_study:
            self.study = create_study(
                direction="maximize", sampler=TPESampler()
            )

        search_space = self._prepare_param_borders(param_borders)
        if (
            self._init_params_in_search_space(search_space)
            and not self._params_tried()
        ):
            self.study.enqueue_trial(self._init_args)

        split_data = self._prepare_split_data(train_dataset, test_dataset)
        objective = self._objective(
            search_space=search_space,
            split_data=split_data,
            recommender=self,
            criterion=self.criterion,
            k=k,
        )

        self.study.optimize(objective, budget)
        best_params = self.study.best_params
        self.set_params(**best_params)
        return best_params

    def _init_params_in_search_space(self, search_space):
        """Check if model params are inside search space"""
        params = self._init_args  # pylint: disable=no-member
        outside_search_space = {}
        for param, value in params.items():
            if param not in search_space:
                continue
            borders = search_space[param]["args"]
            param_type = search_space[param]["type"]

            extra_category = (
                param_type == "categorical" and value not in borders
            )
            param_out_of_bounds = param_type != "categorical" and (
                value < borders[0] or value > borders[1]
            )
            if extra_category or param_out_of_bounds:
                outside_search_space[param] = {
                    "borders": borders,
                    "value": value,
                }

        if outside_search_space:
            self.logger.debug(
                "Model is initialized with parameters outside the search space: %s."
                "Initial parameters will not be evaluated during optimization."
                "Change search spare with 'param_borders' argument if necessary",
                outside_search_space,
            )
            return False
        else:
            return True

    def _prepare_param_borders(
        self, param_borders: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, Dict[str, List[Any]]]:
        """
        Checks if param borders are valid and convert them to a search_space format

        :param param_borders: a dictionary with search grid, where
            key is the parameter name and value is the range of possible values
            ``{param: [low, high]}``.
        :return:
        """
        search_space = deepcopy(self._search_space)
        if param_borders is None:
            return search_space

        for param, borders in param_borders.items():
            self._check_borders(param, borders)
            search_space[param]["args"] = borders

        # Optuna trials should contain all searchable parameters
        # to be able to correctly return best params
        # If used didn't specify some params to be tested optuna still needs to suggest them
        # This part makes sure this suggestion will be constant
        args = self._init_args
        missing_borders = {
            param: args[param]
            for param in search_space
            if param not in param_borders
        }
        for param, value in missing_borders.items():
            if search_space[param]["type"] == "categorical":
                search_space[param]["args"] = [value]
            else:
                search_space[param]["args"] = [value, value]

        return search_space

    def _check_borders(self, param, borders):
        """Raise value error if param borders are not valid"""
        if param not in self._search_space:
            raise ValueError(
                f"Hyper parameter {param} is not defined for {str(self)}"
            )
        if not isinstance(borders, list):
            raise ValueError(f"Parameter {param} borders are not a list")
        if (
            self._search_space[param]["type"] != "categorical"
            and len(borders) != 2
        ):
            raise ValueError(
                f"""
                Hyper parameter {param} is numerical
                 but bounds are not in ([lower, upper]) format
                """
            )

    def _prepare_split_data(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ) -> SplitData:
        """
        This method converts data to spark and packs it into a named tuple to pass into optuna.

        :param train_dataset: train data
        :param test_dataset: test data
        :return: packed PySpark DataFrames
        """
        train = self._filter_dataset_features(train_dataset)
        test = self._filter_dataset_features(test_dataset)
        queries = test_dataset.interactions.select(self.query_column).distinct()
        items = test_dataset.interactions.select(self.item_column).distinct()
        split_data = SplitData(
            train,
            test,
            queries,
            items,
        )
        return split_data

    @staticmethod
    def _filter_dataset_features(
        dataset: Dataset,
    ) -> Dataset:
        """
        Filter features of dataset to match with items and queries of interactions

        :param dataset: dataset with interactions and features
        :return: filtered dataset
        """
        if dataset.query_features is None and dataset.item_features is None:
            return dataset

        query_features = None
        item_features = None
        if dataset.query_features is not None:
            query_features = dataset.query_features.join(
                dataset.interactions.select(
                    dataset.feature_schema.query_id_column
                ).distinct(),
                on=dataset.feature_schema.query_id_column,
            )
        if dataset.item_features is not None:
            item_features = dataset.item_features.join(
                dataset.interactions.select(
                    dataset.feature_schema.item_id_column
                ).distinct(),
                on=dataset.feature_schema.item_id_column,
            )

        return Dataset(
            feature_schema=dataset.feature_schema,
            interactions=dataset.interactions,
            query_features=query_features,
            item_features=item_features,
            check_consistency=False,
            categorical_encoded=False,
        )

    def _fit_wrap(
        self,
        dataset: Dataset,
    ) -> None:
        """
        Wrapper for fit to allow for fewer arguments in a model.

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :return:
        """
        self.query_column = dataset.feature_schema.query_id_column
        self.item_column = dataset.feature_schema.item_id_column
        self.rating_column = dataset.feature_schema.interactions_rating_column
        self.timestamp_column = dataset.feature_schema.interactions_timestamp_column
        self.logger.debug("Starting fit %s", type(self).__name__)
        if dataset.query_features is None:
            queries = dataset.interactions.select(self.query_column).distinct()
        else:
            queries = (
                dataset.interactions.select(self.query_column)
                .union(dataset.query_features.select(self.query_column))
                .distinct()
            )
        if dataset.item_features is None:
            items = dataset.interactions.select(self.item_column).distinct()
        else:
            items = (
                dataset.interactions.select(self.item_column)
                .union(dataset.item_features.select(self.item_column))
                .distinct()
            )
        self.fit_queries = sf.broadcast(queries)
        self.fit_items = sf.broadcast(items)
        self._num_queries = self.fit_queries.count()
        self._num_items = self.fit_items.count()
        self._query_dim_size = (
            self.fit_queries.agg({self.query_column: "max"}).collect()[0][0] + 1
        )
        self._item_dim_size = (
            self.fit_items.agg({self.item_column: "max"}).collect()[0][0] + 1
        )
        self._fit(dataset)

    @abstractmethod
    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        """
        Inner method where model actually fits.

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :return:
        """

    def _filter_seen(
        self, recs: DataFrame, interactions: DataFrame, k: int, queries: DataFrame
    ):
        """
        Filter seen items (presented in interactions) out of the queries' recommendations.
        For each query return from `k` to `k + number of seen by query` recommendations.
        """
        queries_interactions = interactions.join(queries, on=self.query_column)
        self._cache_model_temp_view(queries_interactions, "filter_seen_queries_interactions")
        num_seen = queries_interactions.groupBy(self.query_column).agg(
            sf.count(self.item_column).alias("seen_count")
        )
        self._cache_model_temp_view(num_seen, "filter_seen_num_seen")

        # count maximal number of items seen by queries
        max_seen = 0
        if num_seen.count() > 0:
            max_seen = num_seen.select(sf.max("seen_count")).collect()[0][0]

        # crop recommendations to first k + max_seen items for each query
        recs = recs.withColumn(
            "temp_rank",
            sf.row_number().over(
                Window.partitionBy(self.query_column).orderBy(
                    sf.col(self.rating_column).desc()
                )
            ),
        ).filter(sf.col("temp_rank") <= sf.lit(max_seen + k))

        # leave k + number of items seen by query recommendations in recs
        recs = (
            recs.join(num_seen, on=self.query_column, how="left")
            .fillna(0)
            .filter(sf.col("temp_rank") <= sf.col("seen_count") + sf.lit(k))
            .drop("temp_rank", "seen_count")
        )

        # filter recommendations presented in interactions interactions
        recs = recs.join(
            queries_interactions.withColumnRenamed(self.item_column, "item")
            .withColumnRenamed(self.query_column, "query")
            .select("query", "item"),
            on=(sf.col(self.query_column) == sf.col("query"))
            & (sf.col(self.item_column) == sf.col("item")),
            how="anti",
        ).drop("query", "item")

        return recs

    def _filter_interactions_queries_items_dataframes(
        self,
        dataset: Optional[Dataset],
        k: int,
        queries: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
    ):
        """
        Returns triplet of filtered `dataset`, `queries`, and `items`.
        Filters out cold entities (queries/items) from the `queries`/`items` and `dataset`
        if the model does not predict cold.
        Filters out duplicates from `queries` and `items` dataframes,
        and excludes all columns except `user_idx` and `item_idx`.

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param k: number of recommendations for each user
        :param queries: queries to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all queries from ``dataset``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``dataset``.
            If it contains new items, ``rating`` for them will be ``0``.
        :return: triplet of filtered `dataset`, `queries`, and `items`.
        """
        self.logger.debug("Starting predict %s", type(self).__name__)
        if dataset is not None:
            query_data = queries or dataset.interactions or dataset.query_features or self.fit_queries
            interactions = dataset.interactions
        else:
            query_data = queries or self.fit_queries
            interactions = None

        queries = get_unique_entities(query_data, self.query_column)
        queries, interactions = self._filter_cold_for_predict(queries, interactions, "query")

        item_data = items or self.fit_items
        items = get_unique_entities(item_data, self.item_column)
        items, interactions = self._filter_cold_for_predict(items, interactions, "item")
        num_items = items.count()
        if num_items < k:
            message = f"k = {k} > number of items = {num_items}"
            self.logger.debug(message)

        if dataset is not None:
            dataset = Dataset(
                feature_schema=dataset.feature_schema,
                interactions=interactions,
                query_features=dataset.query_features,
                item_features=dataset.item_features,
                check_consistency=False,
            )
        return dataset, queries, items

    # pylint: disable=too-many-arguments
    def _predict_wrap(
        self,
        dataset: Optional[Dataset],
        k: int,
        queries: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Predict wrapper to allow for fewer parameters in models

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param k: number of recommendations for each user
        :param queries: queries to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all queries from ``interactions``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``interactions``.
            If it contains new items, ``rating`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``interactions``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, rating]``
            or None if `file_path` is provided
        """
        dataset, queries, items = self._filter_interactions_queries_items_dataframes(
            dataset, k, queries, items
        )

        recs = self._predict(
            dataset,
            k,
            queries,
            items,
            filter_seen_items,
        )
        if filter_seen_items and dataset is not None:
            recs = self._filter_seen(recs=recs, interactions=dataset.interactions, queries=queries, k=k)

        recs = get_top_k_recs(recs, k=k, query_column=self.query_column, rating_column=self.rating_column).select(
            self.query_column, self.item_column, self.rating_column
        )

        output = return_recs(recs, recs_file_path)
        self._clear_model_temp_view("filter_seen_queries_interactions")
        self._clear_model_temp_view("filter_seen_num_seen")
        return output

    def _filter_cold_for_predict(
        self,
        main_df: DataFrame,
        interactions_df: Optional[DataFrame],
        entity: str,
    ):
        """
        Filter out cold entities (queries/items) from the `main_df` and `interactions_df_df`
        if the model does not predict cold.
        Warn if cold entities are present in the `main_df`.
        """
        can_predict_cold = self.can_predict_cold_queries if entity == "query" else self.can_predict_cold_items
        fit = self.fit_queries if entity == "query" else self.fit_items
        column = self.query_column if entity == "query" else self.item_column
        if can_predict_cold:
            return main_df, interactions_df

        num_new, main_df = filter_cold(
            main_df, fit, col_name=column
        )
        if num_new > 0:
            self.logger.info(
                "%s model can't predict cold %ss, they will be ignored",
                self,
                entity,
            )
        _, interactions_df = filter_cold(
            interactions_df, fit, col_name=column
        )
        return main_df, interactions_df

    # pylint: disable=too-many-arguments
    @abstractmethod
    def _predict(
        self,
        dataset: Dataset,
        k: int,
        queries: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Inner method where model actually predicts.

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param k: number of recommendations for each user
        :param queries: queries to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all queries from ``interactions``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``interactions``.
            If it contains new items, ``rating`` for them will be ``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``interactions``.
        :return: recommendation dataframe
            ``[user_idx, item_idx, rating]``
        """

    def _get_fit_counts(self, entity: str) -> int:
        num = "_num_queries" if entity == "query" else "_num_items"
        fit = self.fit_queries if entity == "query" else self.fit_items
        if not hasattr(self, num):
            setattr(
                self,
                num,
                fit.count(),
            )
        return getattr(self, num)

    @property
    def queries_count(self) -> int:
        """
        :returns: number of queries the model was trained on
        """
        return self._get_fit_counts("query")

    @property
    def items_count(self) -> int:
        """
        :returns: number of items the model was trained on
        """
        return self._get_fit_counts("items")

    def _get_fit_dims(self, entity: str) -> int:
        dim_size = f"_{entity}_dim_size"
        fit = self.fit_queries if entity == "query" else self.fit_items
        column = self.query_column if entity == "query" else self.item_column
        if not hasattr(self, dim_size):
            setattr(
                self,
                dim_size,
                fit
                .agg({column: "max"})
                .collect()[0][0]
                + 1,
            )
        return getattr(self, dim_size)

    @property
    def _query_dim(self) -> int:
        """
        :returns: dimension of queries matrix (maximal query id + 1)
        """
        return self._get_fit_dims("query")

    @property
    def _item_dim(self) -> int:
        """
        :returns: dimension of items matrix (maximal item idx + 1)
        """
        return self._get_fit_dims("item")

    def _fit_predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        self._fit_wrap(dataset)
        return self._predict_wrap(
            dataset,
            k,
            queries,
            items,
            filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def _predict_pairs_wrap(
        self,
        pairs: DataFrame,
        dataset: Optional[Dataset] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[DataFrame]:
        """
        This method
        1) converts data to spark
        2) removes cold queries and items if model does not predict them
        3) calls inner _predict_pairs method of a model

        :param pairs: query-item pairs to get rating for,
            dataframe containing``[user_idx, item_idx]``.
        :param dataset: train data
            ``[user_idx, item_idx, timestamp, rating]``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached dataframe with columns ``[user_idx, item_idx, rating]``
            or None if `file_path` is provided
        """
        if dataset is not None:
            interactions, query_features, item_features, pairs = [
                convert2spark(df)
                for df in [dataset.interactions, dataset.query_features, dataset.item_features, pairs]
            ]
            if set(pairs.columns) != set([self.item_column, self.query_column]):
                raise ValueError(
                    "pairs must be a dataframe with columns strictly [user_idx, item_idx]"
                )
            pairs, interactions = self._filter_cold_for_predict(pairs, interactions, "query")
            pairs, interactions = self._filter_cold_for_predict(pairs, interactions, "item")

            dataset = Dataset(
                feature_schema=dataset.feature_schema,
                interactions=interactions,
                query_features=query_features,
                item_features=item_features,
            )

        pred = self._predict_pairs(
            pairs=pairs,
            dataset=dataset,
        )

        if k:
            pred = get_top_k(
                dataframe=pred,
                partition_by_col=sf.col(self.query_column),
                order_by_col=[
                    sf.col(self.rating_column).desc(),
                ],
                k=k,
            )

        if recs_file_path is not None:
            pred.write.parquet(path=recs_file_path, mode="overwrite")
            return None

        pred.cache().count()
        return pred

    def _predict_pairs(
        self,
        pairs: DataFrame,
        dataset: Optional[Dataset] = None,
    ) -> DataFrame:
        """
        Fallback method to use in case ``_predict_pairs`` is not implemented.
        Simply joins ``predict`` with given ``pairs``.
        :param pairs: query-item pairs to get rating for,
            dataframe containing``[user_idx, item_idx]``.
        :param dataset: train data
            ``[user_idx, item_idx, timestamp, rating]``.
        """
        message = (
            "native predict_pairs is not implemented for this model. "
            "Falling back to usual predict method and filtering the results."
        )
        self.logger.warning(message)

        queries = pairs.select(self.query_column).distinct()
        items = pairs.select(self.item_column).distinct()
        k = items.count()
        pred = self._predict(
            dataset=dataset,
            k=k,
            queries=queries,
            items=items,
            filter_seen_items=False,
        )

        pred = pred.join(
            pairs.select(self.query_column, self.item_column),
            on=[self.query_column, self.item_column],
            how="inner",
        )
        return pred

    def _get_features_wrap(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Optional[Tuple[DataFrame, int]]:
        if self.query_column not in ids.columns and self.item_column not in ids.columns:
            raise ValueError(f"{self.query_column} or {self.item_column} missing")
        vectors, rank = self._get_features(ids, features)
        return vectors, rank

    # pylint: disable=unused-argument
    def _get_features(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Tuple[Optional[DataFrame], Optional[int]]:
        """
        Get embeddings from model

        :param ids: id ids to get embeddings for Spark DataFrame containing user_idx or item_idx
        :param features: query or item features
        :return: DataFrame with biases and embeddings, and vector size
        """

        self.logger.info(
            "get_features method is not defined for the model %s. Features will not be returned.",
            str(self),
        )
        return None, None

    def _get_nearest_items_wrap(
        self,
        items: Union[DataFrame, Iterable],
        k: int,
        metric: Optional[str] = "cosine_similarity",
        candidates: Optional[Union[DataFrame, Iterable]] = None,
    ) -> Optional[DataFrame]:
        """
        Convert indexes and leave top-k nearest items for each item in `items`.
        """
        items = get_unique_entities(items, self.item_column)
        if candidates is not None:
            candidates = get_unique_entities(candidates, self.item_column)

        nearest_items_to_filter = self._get_nearest_items(
            items=items,
            metric=metric,
            candidates=candidates,
        )

        rel_col_name = metric if metric is not None else "similarity"
        nearest_items = get_top_k(
            dataframe=nearest_items_to_filter,
            partition_by_col=sf.col("item_idx_one"),
            order_by_col=[
                sf.col(rel_col_name).desc(),
                sf.col("item_idx_two").desc(),
            ],
            k=k,
        )

        nearest_items = nearest_items.withColumnRenamed(
            "item_idx_two", "neighbour_item_idx"
        )
        nearest_items = nearest_items.withColumnRenamed(
            "item_idx_one", self.item_column
        )
        return nearest_items

    def _get_nearest_items(
        self,
        items: DataFrame,
        metric: Optional[str] = None,
        candidates: Optional[DataFrame] = None,
    ) -> Optional[DataFrame]:
        raise NotImplementedError(
            f"item-to-item prediction is not implemented for {self}"
        )

    def _params_tried(self):
        """check if current parameters were already evaluated"""
        if self.study is None:
            return False

        params = {
            name: value
            for name, value in self._init_args.items()
            if name in self._search_space
        }
        for trial in self.study.trials:
            if params == trial.params:
                return True

        return False

    def _save_model(self, path: str):
        save_picklable_to_parquet(
            {
                "query_column": self.query_column,
                "item_column": self.item_column,
                "rating_column": self.rating_column,
                "timestamp_column": self.timestamp_column,
            },
            join(path, "params.dump")
        )

    def _load_model(self, path: str):
        loaded_params = load_pickled_from_parquet(join(path, "params.dump"))
        self.query_column = loaded_params.get("query_column")
        self.item_column = loaded_params.get("item_column")
        self.rating_column = loaded_params.get("rating_column")
        self.timestamp_column = loaded_params.get("timestamp_column")


class ItemVectorModel(BaseRecommender):
    """Parent for models generating items' vector representations"""

    can_predict_item_to_item: bool = True
    item_to_item_metrics: List[str] = [
        "euclidean_distance_sim",
        "cosine_similarity",
        "dot_product",
    ]

    @abstractmethod
    def _get_item_vectors(self) -> DataFrame:
        """
        Return dataframe with items' vectors as a
            spark dataframe with columns ``[item_idx, item_vector]``
        """

    def get_nearest_items(
        self,
        items: Union[DataFrame, Iterable],
        k: int,
        metric: Optional[str] = "cosine_similarity",
        candidates: Optional[Union[DataFrame, Iterable]] = None,
    ) -> Optional[DataFrame]:
        """
        Get k most similar items be the `metric` for each of the `items`.

        :param items: spark dataframe or list of item ids to find neighbors
        :param k: number of neighbors
        :param metric: 'euclidean_distance_sim', 'cosine_similarity', 'dot_product'
        :param candidates: spark dataframe or list of items
            to consider as similar, e.g. popular/new items. If None,
            all items presented during model training are used.
        :return: dataframe with the most similar items,
            where bigger value means greater similarity.
            spark-dataframe with columns ``[item_idx, neighbour_item_idx, similarity]``
        """
        if metric not in self.item_to_item_metrics:
            raise ValueError(
                f"Select one of the valid distance metrics: "
                f"{self.item_to_item_metrics}"
            )

        return self._get_nearest_items_wrap(
            items=items,
            k=k,
            metric=metric,
            candidates=candidates,
        )

    def _get_nearest_items(
        self,
        items: DataFrame,
        metric: str = "cosine_similarity",
        candidates: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        Return distance metric value for all available close items filtered by `candidates`.

        :param items: ids to find neighbours, spark dataframe with column ``item_idx``
        :param metric: 'euclidean_distance_sim' calculated as 1/(1 + euclidean_distance),
            'cosine_similarity', 'dot_product'
        :param candidates: items among which we are looking for similar,
            e.g. popular/new items. If None, all items presented during model training are used.
        :return: dataframe with neighbours,
            spark-dataframe with columns ``[item_idx_one, item_idx_two, similarity]``
        """
        dist_function = cosine_similarity
        if metric == "euclidean_distance_sim":
            dist_function = vector_euclidean_distance_similarity
        elif metric == "dot_product":
            dist_function = vector_dot

        items_vectors = self._get_item_vectors()
        left_part = (
            items_vectors.withColumnRenamed(self.item_column, "item_idx_one")
            .withColumnRenamed("item_vector", "item_vector_one")
            .join(
                items.select(sf.col(self.item_column).alias("item_idx_one")),
                on="item_idx_one",
            )
        )

        right_part = items_vectors.withColumnRenamed(
            self.item_column, "item_idx_two"
        ).withColumnRenamed("item_vector", "item_vector_two")

        if candidates is not None:
            right_part = right_part.join(
                candidates.withColumnRenamed(self.item_column, "item_idx_two"),
                on="item_idx_two",
            )

        joined_factors = left_part.join(
            right_part, on=sf.col("item_idx_one") != sf.col("item_idx_two")
        )

        joined_factors = joined_factors.withColumn(
            metric,
            dist_function(
                sf.col("item_vector_one"), sf.col("item_vector_two")
            ),
        )

        similarity_matrix = joined_factors.select(
            "item_idx_one", "item_idx_two", metric
        )

        return similarity_matrix


# pylint: disable=abstract-method
class HybridRecommender(BaseRecommender, ABC):
    """Base class for models that can use extra features"""

    def fit(
        self,
        dataset: Dataset,
    ) -> None:
        """
        Fit a recommendation model

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :return:
        """
        self._fit_wrap(dataset=dataset)

    # pylint: disable=too-many-arguments
    def predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Get recommendations

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param k: number of recommendations for each query
        :param queries: queries to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all queries from ``interactions``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``interactions``.
            If it contains new items, ``rating`` for them will be ``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``interactions``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, rating]``
            or None if `file_path` is provided

        """
        return self._predict_wrap(
            dataset=dataset,
            k=k,
            queries=queries,
            items=items,
            filter_seen_items=filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def fit_predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Fit model and get recommendations

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param k: number of recommendations for each query
        :param queries: queries to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all queries from ``interactions``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``interactions``.
            If it contains new items, ``rating`` for them will be ``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``interactions``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, rating]``
            or None if `file_path` is provided
        """
        return self._fit_predict(
            dataset=dataset,
            k=k,
            queries=queries,
            items=items,
            filter_seen_items=filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def predict_pairs(
        self,
        pairs: DataFrame,
        dataset: Optional[Dataset] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[DataFrame]:
        """
        Get recommendations for specific query-item ``pairs``.
        If a model can't produce recommendation
        for specific pair it is removed from the resulting dataframe.

        :param pairs: dataframe with pairs to calculate rating for, ``[user_idx, item_idx]``.
        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :param k: top-k items for each query from pairs.
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, rating]``
            or None if `file_path` is provided
        """
        return self._predict_pairs_wrap(
            pairs=pairs,
            dataset=dataset,
            recs_file_path=recs_file_path,
            k=k,
        )

    def get_features(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Optional[Tuple[DataFrame, int]]:
        """
        Returns query or item feature vectors as a Column with type ArrayType
        :param ids: Spark DataFrame with unique ids
        :param features: Spark DataFrame with features for provided ids
        :return: feature vectors
            If a model does not have a vector for some ids they are not present in the final result.
        """
        return self._get_features_wrap(ids, features)


# pylint: disable=abstract-method
class Recommender(BaseRecommender, ABC):
    """Usual recommender class for models without features."""

    def fit(self, dataset: Dataset) -> None:
        """
        Fit a recommendation model

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :return:
        """
        self._fit_wrap(dataset=dataset)

    # pylint: disable=too-many-arguments
    def predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Get recommendations

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param k: number of recommendations for each query
        :param queries: queries to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all queries from ``interactions``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``interactions``.
            If it contains new items, ``rating`` for them will be ``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``interactions``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, rating]``
            or None if `file_path` is provided
        """
        return self._predict_wrap(
            dataset=dataset,
            k=k,
            queries=queries,
            items=items,
            filter_seen_items=filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def predict_pairs(
        self,
        pairs: DataFrame,
        dataset: Optional[Dataset] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[DataFrame]:
        """
        Get recommendations for specific query-item ``pairs``.
        If a model can't produce recommendation
        for specific pair it is removed from the resulting dataframe.

        :param pairs: dataframe with pairs to calculate rating for, ``[user_idx, item_idx]``.
        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :param k: top-k items for each query from pairs.
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, rating]``
            or None if `file_path` is provided
        """
        return self._predict_pairs_wrap(
            pairs=pairs,
            dataset=dataset,
            recs_file_path=recs_file_path,
            k=k,
        )

    # pylint: disable=too-many-arguments
    def fit_predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Fit model and get recommendations

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param k: number of recommendations for each query
        :param queries: queries to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all queries from ``interactions``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``interactions``.
            If it contains new items, ``rating`` for them will be ``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``interactions``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, rating]``
            or None if `file_path` is provided
        """
        return self._fit_predict(
            dataset=dataset,
            k=k,
            queries=queries,
            items=items,
            filter_seen_items=filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def get_features(self, ids: DataFrame) -> Optional[Tuple[DataFrame, int]]:
        """
        Returns query or item feature vectors as a Column with type ArrayType

        :param ids: Spark DataFrame with unique ids
        :return: feature vectors.
            If a model does not have a vector for some ids they are not present in the final result.
        """
        return self._get_features_wrap(ids, None)


class QueryRecommender(BaseRecommender, ABC):
    """Base class for models that use query features
    but not item features. ``interactions`` is not required for this class."""

    def fit(
        self,
        dataset: Dataset,
    ) -> None:
        """
        Finds query clusters and calculates item similarity in that clusters.

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :return:
        """
        self._fit_wrap(dataset=dataset)

    # pylint: disable=too-many-arguments
    def predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Get recommendations

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param k: number of recommendations for each query
        :param queries: queries to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all queries from ``interactions``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``interactions``.
            If it contains new items, ``rating`` for them will be ``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``interactions``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, rating]``
            or None if `file_path` is provided
        """
        return self._predict_wrap(
            dataset=dataset,
            k=k,
            filter_seen_items=filter_seen_items,
            queries=queries,
            items=items,
            recs_file_path=recs_file_path,
        )

    def predict_pairs(
        self,
        pairs: DataFrame,
        dataset: Dataset,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[DataFrame]:
        """
        Get recommendations for specific query-item ``pairs``.
        If a model can't produce recommendation
        for specific pair it is removed from the resulting dataframe.

        :param pairs: dataframe with pairs to calculate rating for, ``[user_idx, item_idx]``.
        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :param k: top-k items for each query from pairs.
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, rating]``
            or None if `file_path` is provided
        """
        return self._predict_pairs_wrap(
            pairs=pairs,
            dataset=dataset,
            recs_file_path=recs_file_path,
            k=k,
        )


class NonPersonalizedRecommender(Recommender, ABC):
    """Base class for non-personalized recommenders with popularity statistics."""

    can_predict_cold_queries = True
    can_predict_cold_items = True
    item_popularity: DataFrame
    add_cold_items: bool
    cold_weight: float
    sample: bool
    fill: float
    seed: Optional[int] = None

    def __init__(self, add_cold_items: bool, cold_weight: float):
        self.add_cold_items = add_cold_items
        if 0 < cold_weight <= 1:
            self.cold_weight = cold_weight
        else:
            raise ValueError(
                "`cold_weight` value should be in interval (0, 1]"
            )

    @property
    def _dataframes(self):
        return {"item_popularity": self.item_popularity}

    def _save_model(self, path: str):
        save_picklable_to_parquet(
            {
                "query_column": self.query_column,
                "item_column": self.item_column,
                "rating_column": self.rating_column,
                "timestamp_column": self.timestamp_column,
                "fill": self.fill,
            },
            join(path, "params.dump")
        )

    def _load_model(self, path: str):
        loaded_params = load_pickled_from_parquet(join(path, "params.dump"))
        self.query_column = loaded_params.get("query_column")
        self.item_column = loaded_params.get("item_column")
        self.rating_column = loaded_params.get("rating_column")
        self.timestamp_column = loaded_params.get("timestamp_column")
        self.fill = loaded_params.get("fill")

    def _clear_cache(self):
        if hasattr(self, "item_popularity"):
            self.item_popularity.unpersist()

    @staticmethod
    def _calc_fill(item_popularity: DataFrame, weight: float, rating_column: str) -> float:
        """
        Calculating a fill value a the minimal rating
        calculated during model training multiplied by weight.
        """
        return (
            item_popularity.select(sf.min(rating_column)).collect()[0][0]
            * weight
        )

    @staticmethod
    def _check_rating(dataset: Dataset):
        rating_column = dataset.feature_schema.interactions_rating_column
        vals = dataset.interactions.select(rating_column).where(
            (sf.col(rating_column) != 1) & (sf.col(rating_column) != 0)
        )
        if vals.count() > 0:
            raise ValueError("Rating values in interactions must be 0 or 1")

    def _get_selected_item_popularity(self, items: DataFrame) -> DataFrame:
        """
        Choose only required item from `item_popularity` dataframe
        for further recommendations generation.
        """
        return self.item_popularity.join(
            items,
            on=self.item_column,
            how="right" if self.add_cold_items else "inner",
        ).fillna(value=self.fill, subset=[self.rating_column])

    @staticmethod
    def _calc_max_hist_len(dataset: Dataset, queries: DataFrame) -> int:
        query_column = dataset.feature_schema.query_id_column
        item_column = dataset.feature_schema.item_id_column
        max_hist_len = (
            (
                dataset.interactions.join(queries, on=query_column)
                .groupBy(query_column)
                .agg(sf.countDistinct(item_column).alias("items_count"))
            )
            .select(sf.max("items_count"))
            .collect()[0][0]
        )
        # all queries have empty history
        if max_hist_len is None:
            max_hist_len = 0

        return max_hist_len

    # pylint: disable=too-many-arguments
    def _predict_without_sampling(
        self,
        dataset: Dataset,
        k: int,
        queries: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Regular prediction for popularity-based models,
        top-k most relevant items from `items` are chosen for each query
        """
        selected_item_popularity = self._get_selected_item_popularity(items)
        selected_item_popularity = selected_item_popularity.withColumn(
            "rank",
            sf.row_number().over(
                Window.orderBy(
                    sf.col(self.rating_column).desc(), sf.col(self.item_column).desc()
                )
            ),
        )

        if filter_seen_items and dataset is not None:
            query_to_num_items = (
                dataset.interactions.join(queries, on=self.query_column)
                .groupBy(self.query_column)
                .agg(sf.countDistinct(self.item_column).alias("num_items"))
            )
            queries = queries.join(query_to_num_items, on=self.query_column, how="left")
            queries = queries.fillna(0, "num_items")
            # 'selected_item_popularity' truncation by k + max_seen
            max_seen = queries.select(sf.coalesce(sf.max("num_items"), sf.lit(0))).collect()[0][0]
            selected_item_popularity = selected_item_popularity\
                .filter(sf.col("rank") <= k + max_seen)
            return queries.join(
                selected_item_popularity, on=(sf.col("rank") <= k + sf.col("num_items")), how="left"
            )

        return queries.crossJoin(
            selected_item_popularity.filter(sf.col("rank") <= k)
        ).drop("rank")

    # pylint: disable=too-many-locals
    def _predict_with_sampling(
        self,
        dataset: Dataset,
        k: int,
        queries: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Randomized prediction for popularity-based models,
        top-k items from `items` are sampled for each query based with
        probability proportional to items' popularity
        """
        selected_item_popularity = self._get_selected_item_popularity(items)
        selected_item_popularity = selected_item_popularity.withColumn(
            self.rating_column,
            sf.when(sf.col(self.rating_column) == sf.lit(0.0), 0.1**6).otherwise(
                sf.col(self.rating_column)
            ),
        )

        items_pd = selected_item_popularity.withColumn(
            "probability",
            sf.col(self.rating_column)
            / selected_item_popularity.select(sf.sum(self.rating_column)).first()[0],
        ).toPandas()

        rec_schema = get_rec_schema(self.query_column, self.item_column, self.rating_column)
        if items_pd.shape[0] == 0:
            return State().session.createDataFrame([], rec_schema)

        seed = self.seed
        query_column = self.query_column
        item_column = self.item_column
        rating_column = self.rating_column
        class_name = self.__class__.__name__

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            query_idx = pandas_df[query_column][0]
            cnt = pandas_df["cnt"][0]

            if seed is not None:
                local_rng = default_rng(seed + query_idx)
            else:
                local_rng = default_rng()

            items_positions = local_rng.choice(
                np.arange(items_pd.shape[0]),
                size=cnt,
                p=items_pd["probability"].values,
                replace=False,
            )

            # workaround to unify RandomRec and UCB
            if class_name == "RandomRec":
                rating = 1 / np.arange(1, cnt + 1)
            else:
                rating = items_pd["probability"].values[items_positions]

            return pd.DataFrame(
                {
                    query_column: cnt * [query_idx],
                    item_column: items_pd[item_column].values[items_positions],
                    rating_column: rating,
                }
            )

        if dataset is not None and filter_seen_items:
            recs = (
                dataset.interactions.select(self.query_column, self.item_column)
                .distinct()
                .join(queries, how="right", on=self.query_column)
                .groupby(self.query_column)
                .agg(sf.countDistinct(self.item_column).alias("cnt"))
                .selectExpr(
                    self.query_column,
                    f"LEAST(cnt + {k}, {items_pd.shape[0]}) AS cnt",
                )
            )
        else:
            recs = queries.withColumn("cnt", sf.lit(min(k, items_pd.shape[0])))

        return recs.groupby(self.query_column).applyInPandas(grouped_map, rec_schema)

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        dataset: Dataset,
        k: int,
        queries: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:

        if self.sample:
            return self._predict_with_sampling(
                dataset=dataset,
                k=k,
                queries=queries,
                items=items,
                filter_seen_items=filter_seen_items,
            )
        else:
            return self._predict_without_sampling(
                dataset, k, queries, items, filter_seen_items
            )

    def _predict_pairs(
        self,
        pairs: DataFrame,
        dataset: Optional[Dataset] = None,
    ) -> DataFrame:
        return (
            pairs.join(
                self.item_popularity,
                on=self.item_column,
                how="left" if self.add_cold_items else "inner",
            )
            .fillna(value=self.fill, subset=[self.rating_column])
            .select(self.query_column, self.item_column, self.rating_column)
        )
