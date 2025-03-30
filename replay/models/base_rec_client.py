import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from replay.data.dataset import Dataset
from replay.metrics import NDCG, Metric
from replay.optimization.optuna_objective import MainObjective
from replay.utils import DataFrameLike, PandasDataFrame, PolarsDataFrame, SparkDataFrame
from replay.utils.common import convert2pandas, convert2polars, convert2spark


class DataModelMissmatchError(Exception):
    """Model Implementation can't calculate input data due to missmatch of types"""


class NotFittedModelError(Exception):
    """Model not fitted"""


class BaseRecommenderClient(ABC):
    """
    Abstract base class for recommender clients.
    Provides common functionality for managing attributes and implementation objects.
    Terms:
    Implementation: A real model, that realize a common functions of Recommendation model: fit, predict, fit_predict...
        This model is writen on one of frameworks - ``Pandas``, ``Polars`` or ``Spark``
    Client: an object that contain link to 3 implementations - on Polars, on Pandas, on Spark.
        It also provides convertation from one framework to other (for example - from spark-fitted model to pandas)
        It wraps functions of implementation - usually, when you call ``BaseRecommenderClient.fit()``, inside this
        fit you call ``Client._impl.fit()`` add some functionality, for example - type-checking
    Convertation: Change the ``_impl`` link from one implementation to other. When you call ``to_pandas()``, it creates
        implementation of Pandas type, and Client change its ``_impl`` link to this new implementation
        Also, we converts all DataFrameLike properties to selected framework type
        All other properties are retain their values too
    """

    # These attributes creates dynamicly at the import stage, because their functionality is approximately identical
    attributes_after_fit = (
        "fit_items",
        "fit_queries",
        "query_column",
        "item_column",
        "rating_column",
        "timestamp_column",
        "_num_queries",
        "_num_items",
        "_query_dim_size",
        "_item_dim_size",
    )

    @staticmethod
    def _make_property(attribute_name, has_setter=None) -> property:
        """
        Dynamicly creates a property that controls access to an attribute of the implementation object.

        :param attribute_name: The name of the attribute to control access to.
        :type attribute_name: str
        :param has_setter: A list of attribute names that should have both getters and setters.
        :type has_setter: list[str]
        :return: A property object that can be used as a decorator for a method.
        :rtype: property
        """

        def getter(self) -> Any:
            """
            Retrieves the value of the specified attribute from the implementation object.

            :return: result of calling self._impl.attribute_name
            """
            if self.is_fitted:
                return getattr(self._impl, attribute_name)
            else:
                msg = (
                    f"Class '{self._impl.__class__}' does not have the '{attribute_name}'. "
                    f"If class is NoneType - fit model, before call '{attribute_name}'"
                )
                raise AttributeError(msg)

        def setter(self, value) -> None:
            """
            Sets the value of the specified attribute on the underlying implementation object.

            :param value: The value to set. ``DataFrameLike``, ``str`` or `int`.
            """
            expected_class = (
                tuple(DataFrameLike.__args__)  # just 'isinstance(DataFrameLike)' not supported below py3.10
                if attribute_name.startswith("fit_")
                else str if attribute_name.endswith("_column") else int
            )
            if self.is_fitted and isinstance(value, expected_class):
                setattr(self._impl, attribute_name, value)
            elif not self.is_fitted and isinstance(value, expected_class):
                self._init_when_first_impl_arrived_args.update({attribute_name: value})
            else:
                msg = f"Can't set to '{attribute_name}' value '{value}' in class '{self._impl.__class__}'"
                raise AttributeError(msg)

        if has_setter is not None and attribute_name in has_setter:
            return property(getter, setter)
        else:
            return property(getter)

    # Creates a few properties on the import stage
    attributes_after_fit_with_setter = attributes_after_fit[:2]  # fit_items and fit_queries
    for attr in attributes_after_fit:
        # _make_property is the object, __func__ is needed to call as a function
        locals()[attr] = _make_property.__func__(attr, attributes_after_fit_with_setter)

    def __init__(self) -> None:
        self.__impl = None
        self.is_pandas = False
        self.is_spark = False
        self.is_polars = False
        self._logger = logging.getLogger("replay")
        # This dict needed, because a few properties can set before model is fitted (fitted means, self.__impl is None)
        self._init_when_first_impl_arrived_args = {
            "can_predict_cold_queries": False,
            "can_predict_cold_items": False,
            "_search_space": None,
            "_objective": MainObjective,
            "criterion": None,
            "study": None,
        }

    @property
    def _impl(self):
        """
        Getter for the linked implementation object.

        :return: linked implementation - ``None`` or object of one of 3 classes, contained inside ``_class_map``
        """
        return self.__impl

    @_impl.setter
    def _impl(self, value) -> None:
        """
        Setter of the linked implementation object.

        :param value: One of 3 Implementations objects. Value must be one of 3 classes contained inside ``_class_map``
        """
        if not isinstance(value, tuple(self._class_map.values())):
            msg = f"Model can be one of these classes: {tuple(self._class_map.values())}, not '{type(value)}'"
            raise ValueError(msg)
        self.__impl = value
        realization = (
            "spark"
            if isinstance(value, self._class_map["spark"])
            else "pandas" if isinstance(value, self._class_map["pandas"]) else "polars"
        )
        self._assign_implementation_type(realization)

    @property
    def is_fitted(self) -> bool:
        """
        Checks if the model is fitted.

        :return: True if the model is fitted, otherwise False.
        """
        return self._impl is not None

    @property
    @abstractmethod
    def _class_map(self) -> Dict[Literal["pandas", "polars", "spark"], Any]:
        """
        Map of all available implementation classes (On Spark, Polars and Pandas)

        :return: A dictionary mapping implementation types to corresponding classes.
        """

    def _assign_implementation_type(self, type_of_model: Literal["pandas", "spark", "polars"]):
        """
        Assigns the implementation type based on the given type_of_model.

        :param type_of_model: The type of the implementation to assign.
        """
        if type_of_model not in ["pandas", "spark", "polars"]:
            msg = f"Argument type_of_model can be spark|pandas|polars, not {type_of_model}"
            raise ValueError(msg)
        self.is_pandas = type_of_model == "pandas"
        self.is_spark = type_of_model == "spark"
        self.is_polars = type_of_model == "polars"

    def _get_implementation_type(self) -> Optional[Literal["pandas", "polars", "spark"]]:
        """
        Retrieves the implementation type.

        :return: The implementation type ("spark", "pandas", or "polars"),
            or None if model not fitted or no type is assigned.
        """
        if self.is_spark:
            return "spark"
        if self.is_pandas:
            return "pandas"
        if self.is_polars:
            return "polars"
        return None

    @property
    @abstractmethod
    def _init_args(self) -> Dict[str, Any]:
        """
        Dictionary of the model attributes passed during model initialization.
        Used for model saving and loading

        :return: A dictionary containing the initial arguments and his name as a key.
        """

    @property
    @abstractmethod
    def _dataframes(self) -> Dict[str, DataFrameLike]:
        """
        Dictionary of the model dataframes required for inference.
        Used for model saving and loading
        """

    @property
    def cached_dfs(self) -> Set[SparkDataFrame]:
        """Storage of Spark's queries plan"""
        if hasattr(self._impl, "cached_dfs") and self._get_implementation_type() == "spark":
            return self._impl.cached_dfs
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'cached_dfs' attribute"
            raise AttributeError(msg)

    @property
    def logger(self) -> logging.Logger:
        """
        Return the implementation logger, or use default logger
        """
        if hasattr(self._impl, "logger"):
            return self._impl.logger
        else:
            return self._logger

    @property
    def model(self):
        """
        Return the model that contains inside complex recommendation algotithms
        """
        if hasattr(self._impl, "model"):
            return self._impl.model
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'model' attribute"
            raise AttributeError(msg)

    @model.setter
    def model(self, value):
        """
        Set the model that contains inside complex recommendation algotithms
        """
        if self.is_fitted:
            self._impl.model = value
        elif not self.is_fitted:
            self._init_when_first_impl_arrived_args.update({"model": value})

    @property
    def can_predict_cold_queries(self) -> bool:
        if hasattr(self._impl, "can_predict_cold_queries"):
            return self._impl.can_predict_cold_queries
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'can_predict_cold_queries' attribute"
            raise AttributeError(msg)

    @property
    def can_predict_cold_items(self) -> bool:
        if hasattr(self._impl, "can_predict_cold_items"):
            return self._impl.can_predict_cold_items
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'can_predict_cold_items' attribute"
            raise AttributeError(msg)

    @property
    def _search_space(self):
        # Is setter needed?
        if hasattr(self._impl, "_search_space"):
            return self._impl._search_space
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_search_space' attribute"
            raise AttributeError(msg)

    @property
    def _objective(self):
        # Is setter needed?
        if hasattr(self._impl, "_objective"):
            return self._impl._objective
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_objective' attribute"
            raise AttributeError(msg)

    @property
    def items_count(self) -> int:
        """
        Return counts of items, that model has seen.
        Return value only if the model has been fitted or the ``fit_items`` property has been manually assigned

        :return: number of items the model was trained on
        """
        if self.is_fitted and hasattr(self._impl, "items_count"):
            return self._impl.items_count
        elif not self.is_fitted and "fit_items" in self._init_when_first_impl_arrived_args:
            if isinstance(self._init_when_first_impl_arrived_args["fit_items"], PandasDataFrame):
                return self._init_when_first_impl_arrived_args["fit_items"].shape[0]
            elif isinstance(self._init_when_first_impl_arrived_args["fit_items"], PolarsDataFrame):
                return self._init_when_first_impl_arrived_args["fit_items"].height
            elif isinstance(self._init_when_first_impl_arrived_args["fit_items"], SparkDataFrame):
                return self._init_when_first_impl_arrived_args["fit_items"].count()
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'items_count' attribute"
            raise AttributeError(msg)

    @property
    def queries_count(self) -> int:
        """
        Return counts of queries, that model has seen.
        Return value only if the model has been fitted or the ``fit_queries`` property has been manually assigned

        :return: number of queries the model was trained on
        """
        if self.is_fitted and hasattr(self._impl, "queries_count"):
            return self._impl.queries_count
        elif not self.is_fitted and "fit_queries" in self._init_when_first_impl_arrived_args:
            if isinstance(self._init_when_first_impl_arrived_args["fit_queries"], PandasDataFrame):
                return self._init_when_first_impl_arrived_args["fit_queries"].shape[0]
            elif isinstance(self._init_when_first_impl_arrived_args["fit_queries"], PolarsDataFrame):
                return self._init_when_first_impl_arrived_args["fit_queries"].height
            elif isinstance(self._init_when_first_impl_arrived_args["fit_queries"], SparkDataFrame):
                return self._init_when_first_impl_arrived_args["fit_queries"].count()
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'queries_count' attribute"
            raise AttributeError(msg)

    @property
    def study(self):
        if hasattr(self._impl, "study"):
            return self._impl.study
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'study' attribute"
            raise AttributeError(msg)

    @study.setter
    def study(self, value) -> None:
        if self.is_fitted:
            self._impl.study = value
        else:
            self._init_when_first_impl_arrived_args.update({"study": value})

    @property
    def criterion(self):
        """Metric to use for optimization"""
        if hasattr(self._impl, "criterion"):
            return self._impl.criterion
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'criterion' attribute"
            raise AttributeError(msg)

    @criterion.setter
    def criterion(self, value) -> None:
        if self.is_fitted:
            self._impl.criterion = value
        else:
            self._init_when_first_impl_arrived_args.update({"criterion": value})

    def __str__(self):
        return type(self).__name__

    def _save_model(self, path: str) -> None:
        """
        Method for dump model attributes to disk
        :return:
        """
        self._impl._save_model(path)

    def _load_model(self, path: str):
        """
        Method for loading model attributes from disk
        """
        self._impl._load_model(path)

    def set_params(self, **params: Dict[str, Any]) -> None:
        """
        Set a dict of parameters to implementation

        :param params: dictionary param name - param value
        :return:
        """
        if self.is_fitted and hasattr(self._impl, "set_params"):
            self._impl.set_params(**params)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'set_params()' method "
            raise AttributeError(msg)

    def _clear_cache(self) -> None:
        """
        In fitted implementation of type ``spark`` clear the cached dataframes

        :return:
        """
        if hasattr(self._impl, "_clear_cache") and self._get_implementation_type() == "spark":
            return self._impl._clear_cache
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_clear_cache()' method"
            raise AttributeError(msg)

    def _check_input_for_predict_is_correct(
        self,
        dataset: DataFrameLike,
        queries: Optional[Union[DataFrameLike, Iterable]],
        items: Optional[Union[DataFrameLike, Iterable]],
    ) -> bool:
        """
        Checks if the input data for prediction is correct and compatible with the current implementation type.

        :param dataset: The main dataset to use for prediction.
        :param queries: Queries to make predictions for.
        :param items: Items to consider for predictions.
        :return: True if the input data is correct.
        """
        if (
            (
                self.is_spark
                and (
                    not dataset.is_spark
                    or (
                        queries is not None
                        and not isinstance(queries, (SparkDataFrame, Iterable))
                        or items is not None
                        and not isinstance(items, (SparkDataFrame, Iterable))
                    )
                )
            )
            or (
                self.is_pandas
                and (
                    not dataset.is_pandas
                    or (
                        queries is not None
                        and not isinstance(queries, (PandasDataFrame, Iterable))
                        or items is not None
                        and not isinstance(items, (PandasDataFrame, Iterable))
                    )
                )
            )
            or (
                self.is_polars
                and (
                    not dataset.is_polars
                    or (
                        queries is not None
                        and not isinstance(queries, (PolarsDataFrame, Iterable))
                        or items is not None
                        and not isinstance(items, (PolarsDataFrame, Iterable))
                    )
                )
            )
        ):
            msg = "Model Implementation can't calculate input data due to missmatch of types"
            raise DataModelMissmatchError(msg)
        return True

    def optimize(  # pragma: no cover
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
        if self.is_fitted and hasattr(self._impl, "optimize"):
            return self._impl.optimize(train_dataset, test_dataset, param_borders, criterion, k, budget, new_study)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'optimize()' function "
            raise AttributeError(msg)

    def fit(self, dataset: Dataset):
        """
        Fit a recommendation model. Wraps the linked implementation method ``fit()``

        :param dataset: historical interactions with query/item features
            ``[user_idx, item_idx, timestamp, rating]``
        :return:
        """
        realization = (
            "spark" if dataset.is_spark else "pandas" if dataset.is_pandas else "polars" if dataset.is_polars else None
        )
        if realization is not None:
            new_impl = self._class_map[realization](**self._init_args)
            for attr, value in self._init_when_first_impl_arrived_args.items():
                if not hasattr(new_impl, attr) or value is not None and getattr(new_impl, attr) is None:
                    setattr(new_impl, attr, value)
            new_impl.fit(dataset)
            self._impl = new_impl
            self._assign_implementation_type(realization)

    def fit_predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[DataFrameLike, Iterable]] = None,
        items: Optional[Union[DataFrameLike, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrameLike]:
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
        self.fit(dataset)
        recs = self.predict(dataset, k, queries, items, filter_seen_items, recs_file_path)
        return recs

    def predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[DataFrameLike, Iterable]] = None,
        items: Optional[Union[DataFrameLike, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrameLike]:
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
        if not self.is_fitted:
            raise NotFittedModelError()
        if dataset is None:
            self.logger.warn("There is empty dataset at input of predict")
            return None
        self._check_input_for_predict_is_correct(dataset, queries, items)
        recs = self._impl.predict(dataset, k, queries, items, filter_seen_items, recs_file_path)
        return recs

    def predict_pairs(
        self,
        pairs: DataFrameLike,
        dataset: Optional[Dataset] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[DataFrameLike]:  # Тип данных
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
        if not self.is_fitted:
            raise NotFittedModelError()
        if (
            self.is_spark
            != isinstance(pairs, SparkDataFrame)
            != (dataset.is_spark if dataset is not None else self.is_spark)
            or self.is_pandas
            != isinstance(pairs, PandasDataFrame)
            != (dataset.is_pandas if dataset is not None else self.is_pandas)
            or self.is_polars
            != isinstance(pairs, PolarsDataFrame)
            != (dataset.is_polars if dataset is not None else self.is_spark)
        ):
            msg = "Model Implementation can't calculate input data due to missmatch of types"
            raise DataModelMissmatchError(msg)

        recs = self._impl.predict_pairs(pairs, dataset, recs_file_path, k)
        return recs

    def _predict_proba(
        self, dataset: Dataset, k: int, queries: DataFrameLike, items: DataFrameLike, filter_seen_items: bool = True
    ) -> np.ndarray:
        """
        Inner method where model actually predicts probability estimates.

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
        :return: distribution over items for each user with shape
            ``(n_users, n_items, k)``
            where we have probability for each user to choose item at fixed position(top-k).
        """
        if not self.is_fitted:
            raise NotFittedModelError()
        self._check_input_for_predict_is_correct(dataset, queries, items)

        recs = self._impl._predict_proba(dataset, k, queries, items, filter_seen_items)
        return recs

    def get_features(
        self, ids: DataFrameLike, features: Optional[DataFrameLike] = None
    ) -> Optional[Tuple[DataFrameLike, int]]:
        """
        Get embeddings from model

        :param ids: id ids to get embeddings for Spark DataFrame containing user_idx or item_idx
        :param features: query or item features
        :return: DataFrameLike with biases and embeddings, and vector size
        """
        if features is None:
            # Some of _impl.get_features() have 1 mandatory arg, some have 2 mandatory args
            return self._impl.get_features(ids)
        return self._impl.get_features(ids, features)  # pragma: no cover
        # TODO: test arg 'features', when HybridRecommenderClient will implemented

    def to_spark(self):
        """
        Change the current model implementation to a Spark-based implementation.
        Use the same Client object, doesn't creates new - object id doesn't change
        Keeps all setted attributes and dataframes (converts it to selected type)
        Returns client with link to new implementation inside.
        Needs a fitted model to convert

        :return: The same Client with link to Spark-based implementation.
        :rtype: BaseRecommenderClient
        """
        if not self.is_fitted:
            msg = "Can't convert not fitted model"
            raise NotFittedModelError(msg)
        if self.is_spark:
            return self
        # creates implementation object of class from _class_map
        copy_implementation = self._class_map["spark"](**self._init_args)
        copy_implementation.can_predict_cold_queries = self.can_predict_cold_queries
        copy_implementation.can_predict_cold_items = self.can_predict_cold_items
        copy_implementation._search_space = (
            deepcopy(self._search_space)
            if hasattr(self, "_search_space")
            else self._init_when_first_impl_arrived_args["_search_space"]
        )
        copy_implementation._objective = (
            deepcopy(self._objective)
            if hasattr(self, "_objective")
            else self._init_when_first_impl_arrived_args["_objective"]
        )
        copy_implementation.study = (
            deepcopy(self.study) if hasattr(self, "study") else self._init_when_first_impl_arrived_args["study"]
        )
        copy_implementation.criterion = (
            deepcopy(self.criterion)
            if hasattr(self, "criterion")
            else self._init_when_first_impl_arrived_args["criterion"]
        )
        copy_implementation.query_column = self.query_column
        copy_implementation.item_column = self.item_column
        copy_implementation.rating_column = self.rating_column
        copy_implementation.timestamp_column = self.timestamp_column
        copy_implementation._num_queries = self._num_queries
        copy_implementation._num_items = self._num_items
        copy_implementation._query_dim_size = self._query_dim_size
        copy_implementation._item_dim_size = self._item_dim_size
        copy_implementation.fit_items = convert2spark(self.fit_items)
        copy_implementation.fit_queries = convert2spark(self.fit_queries)
        # change the implementation in our current client
        self._impl = copy_implementation
        self._assign_implementation_type("spark")
        return self

    def to_pandas(self):
        """
        Change the current model implementation to a Pandas-based implementation.
        Use the same Client object, doesn't creates new - object id doesn't change
        Keeps all setted attributes and dataframes (converts it to selected type)
        Returns client with link to new implementation inside.
        Needs a fitted model to convert

        :return: The same Client with link to Pandas-based implementation.
        :rtype: BaseRecommenderClient
        """
        if not self.is_fitted:
            msg = "Can't convert not fitted model"
            raise NotFittedModelError(msg)
        if self.is_pandas:
            return self
        # creates implementation object of class from _class_map
        copy_implementation = self._class_map["pandas"](**self._init_args)
        copy_implementation.can_predict_cold_queries = self.can_predict_cold_queries
        copy_implementation.can_predict_cold_items = self.can_predict_cold_items
        copy_implementation._search_space = (
            deepcopy(self._search_space)
            if hasattr(self, "_search_space")
            else self._init_when_first_impl_arrived_args["_search_space"]
        )
        copy_implementation._objective = (
            deepcopy(self._objective)
            if hasattr(self, "_objective")
            else self._init_when_first_impl_arrived_args["_objective"]
        )
        copy_implementation.study = (
            deepcopy(self.study) if hasattr(self, "study") else self._init_when_first_impl_arrived_args["study"]
        )
        copy_implementation.criterion = (
            deepcopy(self.criterion)
            if hasattr(self, "criterion")
            else self._init_when_first_impl_arrived_args["criterion"]
        )
        copy_implementation.query_column = self.query_column
        copy_implementation.item_column = self.item_column
        copy_implementation.rating_column = self.rating_column
        copy_implementation.timestamp_column = self.timestamp_column
        copy_implementation._num_queries = self._num_queries
        copy_implementation._num_items = self._num_items
        copy_implementation._query_dim_size = self._query_dim_size
        copy_implementation._item_dim_size = self._item_dim_size
        copy_implementation.fit_items = convert2pandas(self.fit_items)
        copy_implementation.fit_queries = convert2pandas(self.fit_queries)
        # change the implementation in our current client
        self._impl = copy_implementation
        self._assign_implementation_type("pandas")
        return self

    def to_polars(self):
        """
        Change the current model implementation to a Polars-based implementation.
        Use the same Client object, doesn't creates new - object id doesn't change
        Keeps all setted attributes and dataframes (converts it to selected type)
        Returns client with link to new implementation inside.
        Needs a fitted model to convert

        :return: The same Client with link to Polars-based implementation.
        :rtype: BaseRecommenderClient
        """
        if not self.is_fitted:
            msg = "Can't convert not fitted model"
            raise NotFittedModelError(msg)
        if self.is_polars:
            return self
        # creates implementation object of class from _class_map
        copy_implementation = self._class_map["polars"](**self._init_args)
        copy_implementation.can_predict_cold_queries = self.can_predict_cold_queries
        copy_implementation.can_predict_cold_items = self.can_predict_cold_items
        copy_implementation._search_space = (
            deepcopy(self._search_space)
            if hasattr(self, "_search_space")
            else self._init_when_first_impl_arrived_args["_search_space"]
        )
        copy_implementation._objective = (
            deepcopy(self._objective)
            if hasattr(self, "_objective")
            else self._init_when_first_impl_arrived_args["_objective"]
        )
        copy_implementation.study = (
            deepcopy(self.study) if hasattr(self, "study") else self._init_when_first_impl_arrived_args["study"]
        )
        copy_implementation.criterion = (
            deepcopy(self.criterion)
            if hasattr(self, "criterion")
            else self._init_when_first_impl_arrived_args["criterion"]
        )
        copy_implementation.query_column = self.query_column
        copy_implementation.item_column = self.item_column
        copy_implementation.rating_column = self.rating_column
        copy_implementation.timestamp_column = self.timestamp_column
        copy_implementation._num_queries = self._num_queries
        copy_implementation._num_items = self._num_items
        copy_implementation._query_dim_size = self._query_dim_size
        copy_implementation._item_dim_size = self._item_dim_size
        copy_implementation.fit_items = convert2polars(self.fit_items)
        copy_implementation.fit_queries = convert2polars(self.fit_queries)
        # change the implementation in our current client
        self._impl = copy_implementation
        self._assign_implementation_type("polars")
        return self


class NonPersonolizedRecommenderClient(BaseRecommenderClient, ABC):
    """Base class for non-personalized recommenders with popularity statistics."""

    _sample: bool = False

    def __init__(self, add_cold_items: bool, cold_weight: float) -> None:
        """
        :param add_cold_items: flag to consider cold items in recommendations building
            if present in `items` parameter of `predict` method
            or `pairs` parameter of `predict_pairs` methods.
            If true, cold items are assigned rating equals to the less relevant item rating
            multiplied by cold_weight and may appear among top-K recommendations.
            Otherwise cold items are filtered out.
            Could be changed after model training by setting the `add_cold_items` attribute.
        : param cold_weight: if `add_cold_items` is True,
            cold items are added with reduced rating.
            The rating for cold items is equal to the rating
            of a least relevant item multiplied by a `cold_weight` value.
            `Cold_weight` value should be in interval (0, 1].
        """
        super().__init__()
        self._init_when_first_impl_arrived_args.update(
            {"can_predict_cold_items": True, "can_predict_cold_queries": True, "seed": None}
        )
        self._add_cold_items = add_cold_items
        if 0 < cold_weight <= 1:
            self._cold_weight = cold_weight
        else:
            msg = "'cold_weight' value should be in interval (0, 1]"
            raise ValueError(msg)
        self._seed = None

    @property
    def _init_args(self) -> Dict[str, Any]:
        if hasattr(self._impl, "_init_args"):
            return self._impl._init_args
        return {
            "add_cold_items": self._add_cold_items,
            "cold_weight": self._cold_weight,
        }

    @property
    def add_cold_items(self) -> bool:
        if self.is_fitted and hasattr(self._impl, "add_cold_items"):
            return self._impl.add_cold_items
        else:
            return self._add_cold_items

    @add_cold_items.setter
    def add_cold_items(self, value: bool) -> None:
        if not isinstance(value, bool):
            msg = f"incorrect type of argument 'value' ({type(value)}). Use bool"
            raise ValueError(msg)
        self._add_cold_items = value
        if self.is_fitted:
            self._impl.add_cold_items = self._add_cold_items
        else:
            self._init_when_first_impl_arrived_args.update({"add_cold_items": value})

    @property
    def cold_weight(self) -> float:
        if self.is_fitted and hasattr(self._impl, "cold_weight"):
            return self._impl.cold_weight
        else:
            return self._cold_weight

    @cold_weight.setter
    def cold_weight(self, value: float) -> None:
        if 0 < value <= 1:
            self._cold_weight = value
            if self.is_fitted:
                self._impl.cold_weight = value
            else:
                self._init_when_first_impl_arrived_args.update({"cold_weight": value})
        else:
            msg = "'cold_weight' value should be float in interval (0, 1]"
            raise ValueError(msg)

    @property
    def item_popularity(self) -> DataFrameLike:
        if hasattr(self._impl, "item_popularity"):
            return self._impl.item_popularity
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'item_popularity' attribute"
            raise AttributeError(msg)

    @item_popularity.setter
    def item_popularity(self, value: DataFrameLike) -> None:
        value_type = (
            "spark"
            if isinstance(value, SparkDataFrame)
            else (
                "pandas"
                if isinstance(value, PandasDataFrame)
                else "polars" if isinstance(value, PolarsDataFrame) else None
            )
        )
        if not self._get_implementation_type() == value_type or value_type is None:
            raise DataModelMissmatchError
        self._impl.item_popularity = value

    @property
    def fill(self) -> float:
        if self.is_fitted and hasattr(self._impl, "fill"):
            return self._impl.fill
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'fill' attribute"
            raise AttributeError(msg)

    @fill.setter
    def fill(self, value) -> None:
        if self.is_fitted:
            self._impl.fill = value
        else:
            msg = f"Can't set to 'fill' value {value} in class '{self._impl.__class__}'"
            raise AttributeError(msg)

    @property
    def sample(self) -> bool:
        if hasattr(self._impl, "sample"):
            return self._impl.sample
        else:
            return self._sample

    @property
    def seed(self) -> int:
        if hasattr(self._impl, "seed"):
            return self._impl.seed
        else:
            return self._seed

    @property
    def _dataframes(self):
        if hasattr(self._impl, "_dataframes"):
            return self._impl._dataframes
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_dataframes' attribute"
            raise AttributeError(msg)

    def get_items_pd(self, items: DataFrameLike) -> pd.DataFrame:
        """
        Function to calculate normalized popularities(in fact, probabilities)
        of given items. Returns pandas DataFrame.
        """
        if (
            self.is_fitted
            and hasattr(self._impl, "get_items_pd")
            and not isinstance(self._impl, self._class_map["pandas"])
        ):
            return self._impl.get_items_pd(items)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'get_items_pd' function"
            raise AttributeError(msg)
