# pylint: disable=too-many-lines
"""
Base abstract classes:
- BaseRecommender - the simplest base class
- Recommender - base class for models that fit on interaction log
- HybridRecommender - base class for models that accept user or item features
- UserRecommender - base class that accepts only user features, but not item features
- NeighbourRec - base class that requires log at prediction time
- ItemVectorModel - class for models which provides items' vectors.
    Implements similar items search.
- NonPersonalizedRecommender - base class for non-personalized recommenders
    with popularity statistics
"""
import collections
import joblib
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
from pyspark.sql.column import Column

from replay.constants import REC_SCHEMA
from replay.metrics import Metric, NDCG
from replay.optuna_objective import SplitData, MainObjective
from replay.session_handler import State
from replay.utils import (
    cache_temp_view,
    convert2spark,
    cosine_similarity,
    drop_temp_view,
    get_top_k,
    get_top_k_recs,
    vector_euclidean_distance_similarity,
    vector_dot,
)


# pylint: disable=too-many-instance-attributes
class BaseRecommender(ABC):
    """Base recommender"""

    model: Any
    _logger: Optional[logging.Logger] = None
    can_predict_cold_users: bool = False
    can_predict_cold_items: bool = False
    _search_space: Optional[
        Dict[str, Union[str, Sequence[Union[str, int, float]]]]
    ] = None
    _objective = MainObjective
    study = None
    fit_users: DataFrame
    fit_items: DataFrame
    _num_users: int
    _num_items: int
    _user_dim_size: int
    _item_dim_size: int
    cached_dfs: Optional[Set] = None

    # pylint: disable=too-many-arguments, too-many-locals, no-member
    def optimize(
        self,
        train: DataFrame,
        test: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        param_borders: Optional[Dict[str, List[Any]]] = None,
        criterion: Metric = NDCG(),
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Searches best parameters with optuna.

        :param train: train data
        :param test: test data
        :param user_features: user features
        :param item_features: item features
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

        split_data = self._prepare_split_data(
            train, test, user_features, item_features
        )
        objective = self._objective(
            search_space=search_space,
            split_data=split_data,
            recommender=self,
            criterion=criterion,
            k=k,
        )

        self.study.optimize(objective, budget)
        best_params = self.study.best_params
        self.set_params(**best_params)
        return best_params

    @property
    @abstractmethod
    def _init_args(self):
        """
        Dictionary of the model attributes passed during model initialization.
        Used for model saving and loading
        """

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
        train: DataFrame,
        test: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> SplitData:
        """
        This method converts data to spark and packs it into a named tuple to pass into optuna.

        :param train: train data
        :param test: test data
        :param user_features: user features
        :param item_features: item features
        :return: packed PySpark DataFrames
        """
        user_features_train, user_features_test = self._train_test_features(
            train, test, user_features, "user_idx"
        )
        item_features_train, item_features_test = self._train_test_features(
            train, test, item_features, "item_idx"
        )
        users = test.select("user_idx").distinct()
        items = test.select("item_idx").distinct()
        split_data = SplitData(
            train,
            test,
            users,
            items,
            user_features_train,
            user_features_test,
            item_features_train,
            item_features_test,
        )
        return split_data

    @property
    def _dataframes(self):
        return {}

    def _save_model(self, path: str):
        pass

    def _load_model(self, path: str):
        pass

    @staticmethod
    def _train_test_features(
        train: DataFrame,
        test: DataFrame,
        features: Optional[DataFrame],
        column: Union[str, Column],
    ) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
        """
        split dataframe with features into two dataframes representing
        features for train and tests subset entities, defined by `column`

        :param train: spark dataframe with the train subset
        :param test: spark dataframe with the train subset
        :param features: spark dataframe with users'/items' features
        :param column: column name to use as a key for join (e.g., user_idx or item_idx)
        :return: features for train and test subsets
        """
        if features is not None:
            features_train = features.join(
                train.select(column).distinct(), on=column
            )
            features_test = features.join(
                test.select(column).distinct(), on=column
            )
        else:
            features_train = None
            features_test = None
        return features_train, features_test

    def set_params(self, **params: Dict[str, Any]) -> None:
        """
        Set model parameters

        :param params: dictionary param name - param value
        :return:
        """
        for param, value in params.items():
            setattr(self, param, value)
        self._clear_cache()

    def __str__(self):
        return type(self).__name__

    def _fit_wrap(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Wrapper for fit to allow for fewer arguments in a model.

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features
            ``[user_idx, timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx, timestamp]`` + feature columns
        :return:
        """
        self.logger.debug("Starting fit %s", type(self).__name__)
        if user_features is None:
            users = log.select("user_idx").distinct()
        else:
            users = (
                log.select("user_idx")
                .union(user_features.select("user_idx"))
                .distinct()
            )
        if item_features is None:
            items = log.select("item_idx").distinct()
        else:
            items = (
                log.select("item_idx")
                .union(item_features.select("item_idx"))
                .distinct()
            )
        self.fit_users = sf.broadcast(users)
        self.fit_items = sf.broadcast(items)
        self._num_users = self.fit_users.count()
        self._num_items = self.fit_items.count()
        self._user_dim_size = (
            self.fit_users.agg({"user_idx": "max"}).collect()[0][0] + 1
        )
        self._item_dim_size = (
            self.fit_items.agg({"item_idx": "max"}).collect()[0][0] + 1
        )
        self._fit(log, user_features, item_features)

    @abstractmethod
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Inner method where model actually fits.

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features
            ``[user_idx, timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx, timestamp]`` + feature columns
        :return:
        """

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

    def _filter_seen(
        self, recs: DataFrame, log: DataFrame, k: int, users: DataFrame
    ):
        """
        Filter seen items (presented in log) out of the users' recommendations.
        For each user return from `k` to `k + number of seen by user` recommendations.
        """
        users_log = log.join(users, on="user_idx")
        self._cache_model_temp_view(users_log, "filter_seen_users_log")
        num_seen = users_log.groupBy("user_idx").agg(
            sf.count("item_idx").alias("seen_count")
        )
        self._cache_model_temp_view(num_seen, "filter_seen_num_seen")

        # count maximal number of items seen by users
        max_seen = 0
        if num_seen.count() > 0:
            max_seen = num_seen.select(sf.max("seen_count")).collect()[0][0]

        # crop recommendations to first k + max_seen items for each user
        recs = recs.withColumn(
            "temp_rank",
            sf.row_number().over(
                Window.partitionBy("user_idx").orderBy(
                    sf.col("relevance").desc()
                )
            ),
        ).filter(sf.col("temp_rank") <= sf.lit(max_seen + k))

        # leave k + number of items seen by user recommendations in recs
        recs = (
            recs.join(num_seen, on="user_idx", how="left")
            .fillna(0)
            .filter(sf.col("temp_rank") <= sf.col("seen_count") + sf.lit(k))
            .drop("temp_rank", "seen_count")
        )

        # filter recommendations presented in interactions log
        recs = recs.join(
            users_log.withColumnRenamed("item_idx", "item")
            .withColumnRenamed("user_idx", "user")
            .select("user", "item"),
            on=(sf.col("user_idx") == sf.col("user"))
            & (sf.col("item_idx") == sf.col("item")),
            how="anti",
        ).drop("user", "item")

        return recs

    # pylint: disable=too-many-arguments
    def _predict_wrap(
        self,
        log: Optional[DataFrame],
        k: int,
        users: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Predict wrapper to allow for fewer parameters in models

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        self.logger.debug("Starting predict %s", type(self).__name__)
        user_data = users or log or user_features or self.fit_users
        users = self._get_ids(user_data, "user_idx")
        users, log = self._filter_cold_for_predict(users, log, "user")

        item_data = items or self.fit_items
        items = self._get_ids(item_data, "item_idx")
        items, log = self._filter_cold_for_predict(items, log, "item")
        num_items = items.count()
        if num_items < k:
            message = f"k = {k} > number of items = {num_items}"
            self.logger.debug(message)

        recs = self._predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )
        if filter_seen_items and log:
            recs = self._filter_seen(recs=recs, log=log, users=users, k=k)

        recs = get_top_k_recs(recs, k=k).select(
            "user_idx", "item_idx", "relevance"
        )
        output = None

        if recs_file_path is not None:
            recs.write.parquet(path=recs_file_path, mode="overwrite")
        else:
            output = recs.cache()
            output.count()

        self._clear_model_temp_view("filter_seen_users_log")
        self._clear_model_temp_view("filter_seen_num_seen")
        return output

    @staticmethod
    def _get_ids(
        log: Union[Iterable, DataFrame],
        column: str,
    ) -> DataFrame:
        """
        Get unique values from ``array`` and put them into dataframe with column ``column``.
        """
        spark = State().session
        if isinstance(log, DataFrame):
            unique = log.select(column).distinct()
        elif isinstance(log, collections.abc.Iterable):
            unique = spark.createDataFrame(
                data=pd.DataFrame(pd.unique(list(log)), columns=[column])
            )
        else:
            raise ValueError(f"Wrong type {type(log)}")
        return unique

    def _filter_cold(
        self, df: Optional[DataFrame], entity: str, suffix: str = "idx"
    ) -> Tuple[int, Optional[DataFrame]]:
        """
        Filter out new ids if the model cannot predict cold users/items.
        Return number of new users/items and filtered dataframe.
        """
        if getattr(self, f"can_predict_cold_{entity}s") or df is None:
            return 0, df

        col_name = f"{entity}_{suffix}"
        num_cold = (
            df.select(col_name)
            .distinct()
            .join(getattr(self, f"fit_{entity}s"), on=col_name, how="anti")
            .count()
        )
        if num_cold == 0:
            return 0, df

        return num_cold, df.join(
            getattr(self, f"fit_{entity}s"), on=col_name, how="inner"
        )

    def _filter_cold_for_predict(
        self,
        main_df: DataFrame,
        log_df: DataFrame,
        entity: str,
        suffix: str = "idx",
    ):
        """
        Filter out cold entities (users/items) from the `main_df` and `log_df`.
        Warn if cold entities are present in the `main_df`.
        """
        num_new, main_df = self._filter_cold(main_df, entity, suffix)
        if num_new > 0:
            self.logger.info(
                "%s model can't predict cold %ss, they will be ignored",
                self,
                entity,
            )
        _, log_df = self._filter_cold(log_df, entity, suffix)
        return main_df, log_df

    # pylint: disable=too-many-arguments
    @abstractmethod
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
        """
        Inner method where model actually predicts.

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :return: recommendation dataframe
            ``[user_idx, item_idx, relevance]``
        """

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: get library logger
        """
        if self._logger is None:
            self._logger = logging.getLogger("replay")
        return self._logger

    def _get_fit_counts(self, entity: str) -> int:
        if not hasattr(self, f"_num_{entity}s"):
            setattr(
                self,
                f"_num_{entity}s",
                getattr(self, f"fit_{entity}s").count(),
            )
        return getattr(self, f"_num_{entity}s")

    @property
    def users_count(self) -> int:
        """
        :returns: number of users the model was trained on
        """
        return self._get_fit_counts("user")

    @property
    def items_count(self) -> int:
        """
        :returns: number of items the model was trained on
        """
        return self._get_fit_counts("item")

    def _get_fit_dims(self, entity: str) -> int:
        if not hasattr(self, f"_{entity}_dim_size"):
            setattr(
                self,
                f"_{entity}_dim_size",
                getattr(self, f"fit_{entity}s")
                .agg({f"{entity}_idx": "max"})
                .collect()[0][0]
                + 1,
            )
        return getattr(self, f"_{entity}_dim_size")

    @property
    def _user_dim(self) -> int:
        """
        :returns: dimension of users matrix (maximal user idx + 1)
        """
        return self._get_fit_dims("user")

    @property
    def _item_dim(self) -> int:
        """
        :returns: dimension of items matrix (maximal item idx + 1)
        """
        return self._get_fit_dims("item")

    def _fit_predict(
        self,
        log: DataFrame,
        k: int,
        users: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        self._fit_wrap(log, user_features, item_features)
        return self._predict_wrap(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def _clear_cache(self):
        """
        Clear spark cache
        """

    def _predict_pairs_wrap(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[DataFrame]:
        """
        This method
        1) converts data to spark
        2) removes cold users and items if model does not predict them
        3) calls inner _predict_pairs method of a model

        :param pairs: user-item pairs to get relevance for,
            dataframe containing``[user_idx, item_idx]``.
        :param log: train data
            ``[user_idx, item_idx, timestamp, relevance]``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        log, user_features, item_features, pairs = [
            convert2spark(df)
            for df in [log, user_features, item_features, pairs]
        ]
        if sorted(pairs.columns) != ["item_idx", "user_idx"]:
            raise ValueError(
                "pairs must be a dataframe with columns strictly [user_idx, item_idx]"
            )
        pairs, log = self._filter_cold_for_predict(pairs, log, "user")
        pairs, log = self._filter_cold_for_predict(pairs, log, "item")

        pred = self._predict_pairs(
            pairs=pairs,
            log=log,
            user_features=user_features,
            item_features=item_features,
        )

        if k:
            pred = get_top_k(
                dataframe=pred,
                partition_by_col=sf.col("user_idx"),
                order_by_col=[
                    sf.col("relevance").desc(),
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
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        Fallback method to use in case ``_predict_pairs`` is not implemented.
        Simply joins ``predict`` with given ``pairs``.
        :param pairs: user-item pairs to get relevance for,
            dataframe containing``[user_idx, item_idx]``.
        :param log: train data
            ``[user_idx, item_idx, timestamp, relevance]``.
        """
        message = (
            "native predict_pairs is not implemented for this model. "
            "Falling back to usual predict method and filtering the results."
        )
        self.logger.warning(message)

        users = pairs.select("user_idx").distinct()
        items = pairs.select("item_idx").distinct()
        k = items.count()
        pred = self._predict(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=False,
        )

        pred = pred.join(
            pairs.select("user_idx", "item_idx"),
            on=["user_idx", "item_idx"],
            how="inner",
        )
        return pred

    def _get_features_wrap(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Optional[Tuple[DataFrame, int]]:
        if "user_idx" not in ids.columns and "item_idx" not in ids.columns:
            raise ValueError("user_idx or item_idx missing")
        vectors, rank = self._get_features(ids, features)
        return vectors, rank

    # pylint: disable=unused-argument
    def _get_features(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Tuple[Optional[DataFrame], Optional[int]]:
        """
        Get embeddings from model

        :param ids: id ids to get embeddings for Spark DataFrame containing user_idx or item_idx
        :param features: user or item features
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
        items = self._get_ids(items, "item_idx")
        if candidates is not None:
            candidates = self._get_ids(candidates, "item_idx")

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
            "item_idx_one", "item_idx"
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
            items_vectors.withColumnRenamed("item_idx", "item_idx_one")
            .withColumnRenamed("item_vector", "item_vector_one")
            .join(
                items.select(sf.col("item_idx").alias("item_idx_one")),
                on="item_idx_one",
            )
        )

        right_part = items_vectors.withColumnRenamed(
            "item_idx", "item_idx_two"
        ).withColumnRenamed("item_vector", "item_vector_two")

        if candidates is not None:
            right_part = right_part.join(
                candidates.withColumnRenamed("item_idx", "item_idx_two"),
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
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """
        Fit a recommendation model

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features
            ``[user_idx, timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx, timestamp]`` + feature columns
        :return:
        """
        self._fit_wrap(
            log=log,
            user_features=user_features,
            item_features=item_features,
        )

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
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Get recommendations

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided

        """
        return self._predict_wrap(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def fit_predict(
        self,
        log: DataFrame,
        k: int,
        users: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Fit model and get recommendations

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._fit_predict(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[DataFrame]:
        """
        Get recommendations for specific user-item ``pairs``.
        If a model can't produce recommendation
        for specific pair it is removed from the resulting dataframe.

        :param pairs: dataframe with pairs to calculate relevance for, ``[user_idx, item_idx]``.
        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param item_features: item features
            ``[item_idx , timestamp]`` + feature columns
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :param k: top-k items for each user from pairs.
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._predict_pairs_wrap(
            pairs=pairs,
            log=log,
            user_features=user_features,
            item_features=item_features,
            recs_file_path=recs_file_path,
            k=k,
        )

    def get_features(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Optional[Tuple[DataFrame, int]]:
        """
        Returns user or item feature vectors as a Column with type ArrayType
        :param ids: Spark DataFrame with unique ids
        :param features: Spark DataFrame with features for provided ids
        :return: feature vectors
            If a model does not have a vector for some ids they are not present in the final result.
        """
        return self._get_features_wrap(ids, features)


# pylint: disable=abstract-method
class Recommender(BaseRecommender, ABC):
    """Usual recommender class for models without features."""

    def fit(self, log: DataFrame) -> None:
        """
        Fit a recommendation model

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :return:
        """
        self._fit_wrap(
            log=log,
            user_features=None,
            item_features=None,
        )

    # pylint: disable=too-many-arguments
    def predict(
        self,
        log: DataFrame,
        k: int,
        users: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Get recommendations

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._predict_wrap(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=None,
            item_features=None,
            filter_seen_items=filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[DataFrame]:
        """
        Get recommendations for specific user-item ``pairs``.
        If a model can't produce recommendation
        for specific pair it is removed from the resulting dataframe.

        :param pairs: dataframe with pairs to calculate relevance for, ``[user_idx, item_idx]``.
        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :param k: top-k items for each user from pairs.
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._predict_pairs_wrap(
            pairs=pairs,
            log=log,
            recs_file_path=recs_file_path,
            k=k,
        )

    # pylint: disable=too-many-arguments
    def fit_predict(
        self,
        log: DataFrame,
        k: int,
        users: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Fit model and get recommendations

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._fit_predict(
            log=log,
            k=k,
            users=users,
            items=items,
            user_features=None,
            item_features=None,
            filter_seen_items=filter_seen_items,
            recs_file_path=recs_file_path,
        )

    def get_features(self, ids: DataFrame) -> Optional[Tuple[DataFrame, int]]:
        """
        Returns user or item feature vectors as a Column with type ArrayType

        :param ids: Spark DataFrame with unique ids
        :return: feature vectors.
            If a model does not have a vector for some ids they are not present in the final result.
        """
        return self._get_features_wrap(ids, None)


class UserRecommender(BaseRecommender, ABC):
    """Base class for models that use user features
    but not item features. ``log`` is not required for this class."""

    def fit(
        self,
        log: DataFrame,
        user_features: DataFrame,
    ) -> None:
        """
        Finds user clusters and calculates item similarity in that clusters.

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param user_features: user features
            ``[user_idx, timestamp]`` + feature columns
        :return:
        """
        self._fit_wrap(log=log, user_features=user_features)

    # pylint: disable=too-many-arguments
    def predict(
        self,
        user_features: DataFrame,
        k: int,
        log: Optional[DataFrame] = None,
        users: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Get recommendations

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._predict_wrap(
            log=log,
            user_features=user_features,
            k=k,
            filter_seen_items=filter_seen_items,
            users=users,
            items=items,
            recs_file_path=recs_file_path,
        )

    def predict_pairs(
        self,
        pairs: DataFrame,
        user_features: DataFrame,
        log: Optional[DataFrame] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[DataFrame]:
        """
        Get recommendations for specific user-item ``pairs``.
        If a model can't produce recommendation
        for specific pair it is removed from the resulting dataframe.

        :param pairs: dataframe with pairs to calculate relevance for, ``[user_idx, item_idx]``.
        :param user_features: user features
            ``[user_idx , timestamp]`` + feature columns
        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :param k: top-k items for each user from pairs.
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        return self._predict_pairs_wrap(
            pairs=pairs,
            log=log,
            user_features=user_features,
            recs_file_path=recs_file_path,
            k=k,
        )


class NeighbourRec(Recommender, ABC):
    """Base class that requires log at prediction time"""

    similarity: Optional[DataFrame]
    can_predict_item_to_item: bool = True
    can_predict_cold_users: bool = True
    can_change_metric: bool = False
    item_to_item_metrics = ["similarity"]
    _similarity_metric = "similarity"

    @property
    def _dataframes(self):
        return {"similarity": self.similarity}

    def _clear_cache(self):
        if hasattr(self, "similarity"):
            self.similarity.unpersist()

    # pylint: disable=missing-function-docstring
    @property
    def similarity_metric(self):
        return self._similarity_metric

    @similarity_metric.setter
    def similarity_metric(self, value):
        if not self.can_change_metric:
            raise ValueError(
                "This class does not support changing similarity metrics"
            )
        if value not in self.item_to_item_metrics:
            raise ValueError(
                f"Select one of the valid metrics for predict: "
                f"{self.item_to_item_metrics}"
            )
        self._similarity_metric = value

    def _predict_pairs_inner(
        self,
        log: DataFrame,
        filter_df: DataFrame,
        condition: Column,
        users: DataFrame,
    ) -> DataFrame:
        """
        Get recommendations for all provided users
        and filter results with ``filter_df`` by ``condition``.
        It allows to implement both ``predict_pairs`` and usual ``predict``@k.

        :param log: historical interactions, DataFrame
            ``[user_idx, item_idx, timestamp, relevance]``.
        :param filter_df: DataFrame use to filter items:
            ``[item_idx_filter]`` or ``[user_idx_filter, item_idx_filter]``.
        :param condition: condition used for inner join with ``filter_df``
        :param users: users to calculate recommendations for
        :return: DataFrame ``[user_idx, item_idx, relevance]``
        """
        if log is None:
            raise ValueError(
                "log is not provided, but it is required for prediction"
            )

        recs = (
            log.join(users, how="inner", on="user_idx")
            .join(
                self.similarity,
                how="inner",
                on=sf.col("item_idx") == sf.col("item_idx_one"),
            )
            .join(
                filter_df,
                how="inner",
                on=condition,
            )
            .groupby("user_idx", "item_idx_two")
            .agg(sf.sum(self.similarity_metric).alias("relevance"))
            .withColumnRenamed("item_idx_two", "item_idx")
        )
        return recs

    # pylint: disable=too-many-arguments
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

        return self._predict_pairs_inner(
            log=log,
            filter_df=items.withColumnRenamed("item_idx", "item_idx_filter"),
            condition=sf.col("item_idx_two") == sf.col("item_idx_filter"),
            users=users,
        )

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:

        if log is None:
            raise ValueError(
                "log is not provided, but it is required for prediction"
            )

        return self._predict_pairs_inner(
            log=log,
            filter_df=(
                pairs.withColumnRenamed(
                    "user_idx", "user_idx_filter"
                ).withColumnRenamed("item_idx", "item_idx_filter")
            ),
            condition=(sf.col("user_idx") == sf.col("user_idx_filter"))
            & (sf.col("item_idx_two") == sf.col("item_idx_filter")),
            users=pairs.select("user_idx").distinct(),
        )

    def get_nearest_items(
        self,
        items: Union[DataFrame, Iterable],
        k: int,
        metric: Optional[str] = None,
        candidates: Optional[Union[DataFrame, Iterable]] = None,
    ) -> DataFrame:
        """
        Get k most similar items be the `metric` for each of the `items`.

        :param items: spark dataframe or list of item ids to find neighbors
        :param k: number of neighbors
        :param metric: metric is not used to find neighbours in NeighbourRec,
            the parameter is ignored
        :param candidates: spark dataframe or list of items
            to consider as similar, e.g. popular/new items. If None,
            all items presented during model training are used.
        :return: dataframe with the most similar items an distance,
            where bigger value means greater similarity.
            spark-dataframe with columns ``[item_idx, neighbour_item_idx, similarity]``
        """

        if metric is not None:
            self.logger.debug(
                "Metric is not used to determine nearest items in %s model",
                str(self),
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
        metric: Optional[str] = None,
        candidates: Optional[DataFrame] = None,
    ) -> DataFrame:

        similarity_filtered = self.similarity.join(
            items.withColumnRenamed("item_idx", "item_idx_one"),
            on="item_idx_one",
        )

        if candidates is not None:
            similarity_filtered = similarity_filtered.join(
                candidates.withColumnRenamed("item_idx", "item_idx_two"),
                on="item_idx_two",
            )

        return similarity_filtered.select(
            "item_idx_one",
            "item_idx_two",
            "similarity" if metric is None else metric,
        )


class NonPersonalizedRecommender(Recommender, ABC):
    """Base class for non-personalized recommenders with popularity statistics."""

    can_predict_cold_users = True
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
        joblib.dump({"fill": self.fill}, join(path))

    def _load_model(self, path: str):
        self.fill = joblib.load(join(path))["fill"]

    def _clear_cache(self):
        if hasattr(self, "item_popularity"):
            self.item_popularity.unpersist()

    @staticmethod
    def _calc_fill(item_popularity: DataFrame, weight: float) -> float:
        """
        Calculating a fill value a the minimal relevance
        calculated during model training multiplied by weight.
        """
        return (
            item_popularity.select(sf.min("relevance")).collect()[0][0]
            * weight
        )

    @staticmethod
    def _check_relevance(log: DataFrame):

        vals = log.select("relevance").where(
            (sf.col("relevance") != 1) & (sf.col("relevance") != 0)
        )
        if vals.count() > 0:
            raise ValueError("Relevance values in log must be 0 or 1")

    def _get_selected_item_popularity(self, items: DataFrame) -> DataFrame:
        """
        Choose only required item from `item_popularity` dataframe
        for further recommendations generation.
        """
        return self.item_popularity.join(
            items,
            on="item_idx",
            how="right" if self.add_cold_items else "inner",
        ).fillna(value=self.fill, subset=["relevance"])

    @staticmethod
    def _calc_max_hist_len(log: DataFrame, users: DataFrame) -> int:
        max_hist_len = (
            (
                log.join(users, on="user_idx")
                .groupBy("user_idx")
                .agg(sf.countDistinct("item_idx").alias("items_count"))
            )
            .select(sf.max("items_count"))
            .collect()[0][0]
        )
        # all users have empty history
        if max_hist_len is None:
            max_hist_len = 0

        return max_hist_len

    # pylint: disable=too-many-arguments
    def _predict_without_sampling(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Regular prediction for popularity-based models,
        top-k most relevant items from `items` are chosen for each user
        """
        selected_item_popularity = self._get_selected_item_popularity(items)
        selected_item_popularity = selected_item_popularity.withColumn(
            "rank",
            sf.row_number().over(
                Window.orderBy(
                    sf.col("relevance").desc(), sf.col("item_idx").desc()
                )
            ),
        )

        max_hist_len = (
            self._calc_max_hist_len(log, users)
            if filter_seen_items and log is not None
            else 0
        )

        return users.crossJoin(
            selected_item_popularity.filter(sf.col("rank") <= k + max_hist_len)
        ).drop("rank")

    def _predict_with_sampling(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        Randomized prediction for popularity-based models,
        top-k items from `items` are sampled for each user based with
        probability proportional to items' popularity
        """
        selected_item_popularity = self._get_selected_item_popularity(items)
        selected_item_popularity = selected_item_popularity.withColumn(
            "relevance",
            sf.when(sf.col("relevance") == sf.lit(0.0), 0.1**6).otherwise(
                sf.col("relevance")
            ),
        )

        items_pd = selected_item_popularity.withColumn(
            "probability",
            sf.col("relevance")
            / selected_item_popularity.select(sf.sum("relevance")).first()[0],
        ).toPandas()

        if items_pd.shape[0] == 0:
            return State().session.createDataFrame([], REC_SCHEMA)

        seed = self.seed
        class_name = self.__class__.__name__

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            user_idx = pandas_df["user_idx"][0]
            cnt = pandas_df["cnt"][0]

            if seed is not None:
                local_rng = default_rng(seed + user_idx)
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
                relevance = 1 / np.arange(1, cnt + 1)
            else:
                relevance = items_pd["probability"].values[items_positions]

            return pd.DataFrame(
                {
                    "user_idx": cnt * [user_idx],
                    "item_idx": items_pd["item_idx"].values[items_positions],
                    "relevance": relevance,
                }
            )

        if log is not None and filter_seen_items:
            recs = (
                log.select("user_idx", "item_idx")
                .distinct()
                .join(users, how="right", on="user_idx")
                .groupby("user_idx")
                .agg(sf.countDistinct("item_idx").alias("cnt"))
                .selectExpr(
                    "user_idx",
                    f"LEAST(cnt + {k}, {items_pd.shape[0]}) AS cnt",
                )
            )
        else:
            recs = users.withColumn("cnt", sf.lit(min(k, items_pd.shape[0])))

        return recs.groupby("user_idx").applyInPandas(grouped_map, REC_SCHEMA)

    # pylint: disable=too-many-arguments
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

        if self.sample:
            return self._predict_with_sampling(
                log=log,
                k=k,
                users=users,
                items=items,
                filter_seen_items=filter_seen_items,
            )
        else:
            return self._predict_without_sampling(
                log, k, users, items, filter_seen_items
            )

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        return (
            pairs.join(
                self.item_popularity,
                on="item_idx",
                how="left" if self.add_cold_items else "inner",
            )
            .fillna(value=self.fill, subset=["relevance"])
            .select("user_idx", "item_idx", "relevance")
        )
