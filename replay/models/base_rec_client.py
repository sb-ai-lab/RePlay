from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from replay.data.dataset import Dataset
from replay.metrics import NDCG, Metric
from replay.utils import DataFrameLike, PandasDataFrame, PolarsDataFrame, SparkDataFrame
from replay.utils.common import convert2pandas, convert2polars, convert2spark


class DataModelMissmatchError(Exception):
    """Model Implementation can't calculate input data due to missmatch of types"""


class NotFittedModelError(Exception):
    """Model not fitted"""


class BaseRecommenderClient(ABC):
    def __init__(self):
        self.is_fitted = False
        self.is_pandas = False
        self.is_spark = False
        self.is_polars = False

    @property
    @abstractmethod
    def _class_map(self):
        """Map of all implementations (Usually - Spark, Polars and Pandas)"""

    def _assign_realization_type(self, type_of_model: Literal["pandas", "spark", "polars"]):
        if type_of_model not in ["pandas", "spark", "polars"]:
            msg = f"Argument type_of_model can be spark|pandas|polars, not {type_of_model}"
            raise ValueError(msg)
        self.is_pandas = type_of_model == "pandas"
        self.is_spark = type_of_model == "spark"
        self.is_polars = type_of_model == "polars"

    def _get_realization_type(self) -> Optional[str]:
        """
        :returns: Stored dataframe type.
        """
        if self.is_spark:
            return "spark"
        if self.is_pandas:
            return "pandas"
        if self.is_polars:
            return "polars"
        msg = "Model type is not setted"
        self.logger.warning(msg)
        return None

    def _get_all_attributes_or_functions(self):
        if self._impl is None:
            return []
        cls = self._impl.__class__
        all_params = []
        all_params.extend(dir(self._impl))
        all_params.extend(self._impl.__dict__)
        all_params.extend(getattr(cls, "__annotations__", {}))
        all_params.extend(dir(cls))
        return list(set(all_params))

    @property
    @abstractmethod
    def _impl(self):
        """Implementation of model on Spark, Polars or Pandas"""

    @property
    @abstractmethod
    def _init_args(self):
        """
        Dictionary of the model attributes passed during model initialization.
        Used for model saving and loading
        """

    @property
    @abstractmethod
    def _dataframes(self) -> Dict:
        """
        Dictionary of the model dataframes required for inference.
        Used for model saving and loading
        """

    @property
    def cached_dfs(self):
        """Storage of Spark's queries plan"""
        if hasattr(self._impl, "cached_dfs") and self._get_realization_type() == "spark":
            return self._impl.cached_dfs
        elif "cached_dfs" in self._get_all_attributes_or_functions():
            msg = "Attribute 'cached_dfs' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'cached_dfs' attribute"
            raise AttributeError()

    @property
    def fit_items(self):
        """Column of fitted items in model"""
        if hasattr(self._impl, "fit_items"):
            return self._impl.fit_items
        elif "fit_items" in self._get_all_attributes_or_functions():
            msg = "Attribute 'fit_items' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'fit_items' attribute"
            raise AttributeError(msg)

    @fit_items.setter
    def fit_items(self, value):
        """Column of fitted items in model"""
        if isinstance(value, DataFrameLike):
            self._impl.fit_items = value
        else:
            msg = f"Can't set to 'fit_items' value {value} in class '{self._impl.__class__}'"
            raise AttributeError(msg)

    @property
    def fit_queries(self):
        """Column of fitted queries in model. Usually, it's column of user_ids"""
        if hasattr(self._impl, "fit_queries"):  # add the required attribute setting
            return self._impl.fit_queries
        elif "fit_queries" in self._get_all_attributes_or_functions():
            msg = "Attribute 'fit_queries' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'fit_queries' attribute"
            raise AttributeError(msg)

    @fit_queries.setter
    def fit_queries(self, value):
        """Column of fitted queries in model"""
        if isinstance(value, DataFrameLike):
            self._impl.fit_queries = value
        else:
            msg = f"Can't set to 'fit_queries' value ** {value} ** in class '{self._impl.__class__}'"
            raise AttributeError(msg)

    @property
    def queries_count(self):
        if hasattr(self._impl, "queries_count"):
            return self._impl.queries_count
        elif "queries_count" in self._get_all_attributes_or_functions():
            msg = "Attribute 'queries_count' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'queries_count' attribute"
            raise AttributeError(msg)

    @property
    def items_count(self):
        if hasattr(self._impl, "items_count"):
            self.logger.warning("Converting big dataframes from spark to pandas can cause OOM error.")
            return self._impl.items_count
        elif "items_count" in self._get_all_attributes_or_functions():
            msg = "Attribute 'items_count' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'items_count' attribute"
            raise AttributeError(msg)

    @property
    def logger(self):
        """
        :returns: get library logger
        """
        if hasattr(self._impl, "logger"):
            return self._impl.logger
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'logger' attribute"
            raise AttributeError(msg)

    @property
    def model(self):
        if hasattr(self._impl, "model"):
            return self._impl.model
        elif "model" in self._get_all_attributes_or_functions():
            msg = "Attribute 'model' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'model' attribute"
            raise AttributeError(msg)

    @property
    def can_predict_cold_queries(self):
        if hasattr(self._impl, "can_predict_cold_queries"):
            return self._impl.can_predict_cold_queries
        elif "can_predict_cold_queries" in self._get_all_attributes_or_functions():
            msg = "Attribute 'can_predict_cold_queries' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'can_predict_cold_queries' attribute"
            raise AttributeError(msg)

    @property
    def can_predict_cold_items(self):
        if hasattr(self._impl, "can_predict_cold_items"):
            return self._impl.can_predict_cold_items
        elif "can_predict_cold_items" in self._get_all_attributes_or_functions():
            msg = "Attribute 'can_predict_cold_items' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'can_predict_cold_items' attribute"
            raise AttributeError(msg)

    @property
    def _search_space(self):
        if hasattr(self._impl, "_search_space"):  # add the required attribute setting
            return self._impl._search_space
        elif "_search_space" in self._get_all_attributes_or_functions():
            msg = "Attribute '_search_space' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_search_space' attribute"
            raise AttributeError(msg)

    @property
    def _objective(self):
        if hasattr(self._impl, "_objective"):
            return self._impl._objective
        elif "_objective" in self._get_all_attributes_or_functions():
            msg = "Attribute '_objective' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_objective' attribute"
            raise AttributeError(msg)

    @property
    def _study(self):
        if hasattr(self._impl, "_study"):
            return self._impl._study
        elif "_study" in self._get_all_attributes_or_functions():
            msg = "Attribute '_study' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_study' attribute"
            raise AttributeError(msg)

    @property
    def _criterion(self):
        if hasattr(self._impl, "_criterion"):
            return self._impl._criterion
        elif "_criterion" in self._get_all_attributes_or_functions():
            msg = "Attribute '_criterion' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_criterion' attribute"
            raise AttributeError(msg)

    @property
    def query_column(self):
        if hasattr(self._impl, "query_column"):  # add the required attribute setting
            return self._impl.query_column
        elif "_criterion" in self._get_all_attributes_or_functions():
            msg = "Attribute 'query_column' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'query_column' attribute"
            raise AttributeError(msg)

    @property
    def item_column(self):
        if hasattr(self._impl, "item_column"):
            return self._impl.item_column
        elif "_criterion" in self._get_all_attributes_or_functions():
            msg = "Attribute 'item_column' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'item_column' attribute"
            raise AttributeError(msg)

    @property
    def rating_column(self):
        if hasattr(self._impl, "rating_column"):
            return self._impl.rating_column
        elif "_criterion" in self._get_all_attributes_or_functions():
            msg = "Attribute 'rating_column' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'rating_column' attribute"
            raise AttributeError(msg)

    @property
    def timestamp_column(self):
        if hasattr(self._impl, "timestamp_column"):
            return self._impl.timestamp_column
        elif "_criterion" in self._get_all_attributes_or_functions():
            msg = "Attribute 'timestamp_column' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'timestamp_column' attribute"
            raise AttributeError(msg)

    @property
    def _num_queries(self):
        if hasattr(self._impl, "_num_queries"):
            return self._impl._num_queries
        elif "_criterion" in self._get_all_attributes_or_functions():
            msg = "Attribute '_num_queries' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_num_queries' attribute"
            raise AttributeError(msg)

    @property
    def _num_items(self):
        if hasattr(self._impl, "_num_items"):
            return self._impl._num_items
        elif "_criterion" in self._get_all_attributes_or_functions():
            msg = "Attribute '_num_items' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_num_items' attribute"
            raise AttributeError(msg)

    @property
    def _query_dim_size(self):
        if hasattr(self._impl, "_query_dim_size"):
            return self._impl._query_dim_size
        elif "_criterion" in self._get_all_attributes_or_functions():
            msg = "Attribute '_query_dim_size' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_query_dim_size' attribute"
            raise AttributeError(msg)

    @property
    def _item_dim_size(self):
        if hasattr(self._impl, "_item_dim_size"):
            return self._impl._item_dim_size
        elif "_criterion" in self._get_all_attributes_or_functions():
            msg = "Attribute '_item_dim_size' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_item_dim_size' attribute"
            raise AttributeError(msg)

    @property 
    def _before_fit_attributes(self):
        return {
            "can_predict_cold_queries" : self.can_predict_cold_queries,
            "can_predict_cold_items" : self.can_predict_cold_items,
            "_search_space" : deepcopy(self._search_space) if hasattr(self, "_search_space") else None,  # Нужен ли для него property, либо забирать через self._impl
            "_objective" : deepcopy(self._objective) if hasattr(self, "_objective") else None,  # Нужен ли для него property, либо забирать через self._impl
            "_study" : deepcopy(self._study) if hasattr(self, "_study") else None, # Нужен ли для него property, либо забирать через self._impl
            "_criterion" : deepcopy(self._criterion) if hasattr(self, "_criterion") else None, # TODO: # Нужен ли для него property, либо забирать через self._impl
            # TODO: # copy_implementation._init_args = deepcopy(self._init_args
            # )# TODO: Нужно ли здесь вообще копировать init_args и _dataframes
        }
        
    @property
    def _after_fit_attributes(self):
        if self.is_fitted:
            return {
                "query_column": self.query_column,
                "item_column" : self.item_column,
                "rating_column" : self.rating_column,
                "timestamp_column" : self.timestamp_column,
                "_num_queries" : self._num_queries,
                "_num_items" : self._num_items,
                "_query_dim_size" : self._query_dim_size,
                "_item_dim_size" : self._item_dim_size
            }
        return None


    def __str__(self):
        return type(self).__name__

    def save_model(self, path: str) -> None:
        """
        Method for dump model attributes to disk
        """
        self._impl._save_model(path)

    def load_model(self, path: str) -> None:
        """
        Method for loading model attributes from disk
        """
        self._impl._load_model(path)

    def set_params(self, **params: Dict[str, Any]) -> None:
        """
        Set model parameters

        :param params: dictionary param name - param value
        :return:
        """
        if hasattr(self._impl, "logger"):
            self._impl.set_params(**params)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'set_params()' function "
            raise AttributeError(msg)

    def _clear_cache(self):
        """Clear the cache in spark realization"""
        if hasattr(self._impl, "_clear_cache") and self._get_realization_type() == "spark":
            return self._impl._clear_cache
        elif "_clear_cache" in self._get_all_attributes_or_functions():
            msg = "Attribute 'cached_dfs' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'cached_dfs' attribute"
            raise AttributeError(msg)

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
        if hasattr(self._impl, "optimize"):
            return self._impl.optimize(train_dataset, test_dataset, param_borders, criterion, k, budget, new_study)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'optimize()' function "
            raise AttributeError(msg)

    def fit(self, dataset):
        """_RecommenderCommonsSparkImpl._init_args"""
        realization = (
            "spark" if dataset.is_spark else "pandas" if dataset.is_pandas else "polars" if dataset.is_polars else None
        )
        self._assign_realization_type(realization)
        if dataset.is_spark or dataset.is_pandas or dataset.is_polars:
            self._impl = self._class_map[realization](**self._init_args)
        else:
            msg = "Model Implementation can't calculate input data due to missmatch of types"
            raise DataModelMissmatchError(msg)
        self._impl.fit(dataset)
        self.is_fitted = True

    def fit_predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[SparkDataFrame, Iterable]] = None,
        items: Optional[Union[SparkDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrameLike]:
        """_RecommenderCommonsSparkImpl._init_args"""
        realization = (
            "spark" if dataset.is_spark else "pandas" if dataset.is_pandas else "polars" if dataset.is_polars else None
        )
        self._assign_realization_type(realization)
        if (
            self.is_spark != dataset.is_spark
            or self.is_pandas != dataset.is_pandas
            or self.is_polars != dataset.is_polars
        ):
            msg = "Model Implementation can't calculate input data due to missmatch of types"
            raise DataModelMissmatchError(msg)

        if self._impl is None and (dataset.is_spark or dataset.is_pandas or dataset.is_polars):
            self._impl = self._class_map[realization](**self._init_args)
        self._impl.fit(dataset)
        self.is_fitted = True
        recs = self._impl.predict(dataset, k, queries, items, filter_seen_items, recs_file_path)
        return recs

    def predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[SparkDataFrame, Iterable]] = None,
        items: Optional[Union[SparkDataFrame, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrameLike]:
        """_RecommenderCommonsSparkImpl._init_args"""
        if dataset is None or dataset.interactions is None:
            self.logger.warn("There is empty dataset at input of predict")
            return None
        if not self.is_fitted:
            raise NotFittedModelError()
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

        recs = self._impl.predict(dataset, k, queries, items, filter_seen_items, recs_file_path)
        return recs

    def predict_pairs(
        self,
        pairs: SparkDataFrame,
        dataset: Optional[Dataset] = None,
        recs_file_path: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Optional[SparkDataFrame]:  # Тип данных
        """_RecommenderCommonsSparkImpl._init_args"""
        if not self.is_fitted:
            raise NotFittedModelError()
        if (
            self.is_spark != dataset.is_spark
            or self.is_pandas != dataset.is_pandas
            or self.is_polars != dataset.is_polars
        ):
            msg = "Model Implementation can't calculate input data due to missmatch of types"
            raise DataModelMissmatchError(msg)

        recs = self._impl.predict_pairs(pairs, dataset, recs_file_path, k)
        return recs

    def _predict_proba(
        self, dataset: Dataset, k: int, queries: SparkDataFrame, items: SparkDataFrame, filter_seen_items: bool = True
    ) -> np.ndarray:
        """
        Inner method where model actually predicts probability estimates.

        Mainly used in ```OBPOfflinePolicyLearner```.

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
        if (
            (
                self.is_spark
                and (
                    not dataset.is_spark
                    or not isinstance(queries, SparkDataFrame)
                    or not isinstance(items, SparkDataFrame)
                )
            )
            or (
                self.is_pandas
                and (
                    not dataset.is_pandas
                    or not isinstance(queries, PandasDataFrame)
                    or not isinstance(items, PandasDataFrame)
                )
            )
            or (
                self.is_polars
                and (
                    not dataset.is_polars
                    or not isinstance(queries, PolarsDataFrame)
                    or not isinstance(items, PolarsDataFrame)
                )
            )
        ):
            msg = "Model Implementation can't calculate input data due to missmatch of types"
            raise DataModelMissmatchError(msg)

        recs = self._impl._predict_proba(dataset, k, queries, items, filter_seen_items)
        return recs

    def get_features(
        self, ids: SparkDataFrame, features: Optional[SparkDataFrame] = None
    ) -> Optional[Tuple[SparkDataFrame, int]]:
        """
        Returns query or item feature vectors as a Column with type ArrayType

        :param ids: Spark DataFrame with unique ids
        :return: feature vectors.
            If a model does not have a vector for some ids they are not present in the final result.
        """
        if features is not None:
            # Some of _impl.get_features() have 1 arg, some have 2 args
            return self._impl.get_features(ids)
        return self._impl.get_features(ids, features)

    def _copy_base_params_to_new_model(self, copy_implementation):
        copy_implementation.can_predict_cold_queries = self.can_predict_cold_queries
        copy_implementation.can_predict_cold_items = self.can_predict_cold_items
        copy_implementation._search_space = deepcopy(
            self._search_space
        )  # Нужен ли для него property, либо забирать через self._impl
        copy_implementation._objective = deepcopy(
            self._objective
        )  # Нужен ли для него property, либо забирать через self._impl
        copy_implementation._study = (
            deepcopy(self._study) if hasattr(self, "_study") else None
        )  # Нужен ли для него property, либо забирать через self._impl
        copy_implementation._criterion = (
            deepcopy(self._criterion) if hasattr(self, "_criterion") else None
        )  # TODO: # Нужен ли для него property, либо забирать через self._impl
        # TODO: # copy_implementation._init_args = deepcopy(self._init_args
        # )# Нужно ли здесь вообще копировать init_args и _dataframes
        if self.is_fitted:
            for name, value in self._after_fit_attributes.items():
                setattr(copy_implementation, name, value)
        return copy_implementation

    def to_spark(self):
        if self.is_spark:
            return self
        copy_implementation = self._class_map["spark"](**self._init_args)
        self._assign_realization_type("spark")
        copy_implementation = self._copy_base_params_to_new_model(copy_implementation)
        if self.is_fitted:
            copy_implementation.fit_items = convert2spark(self.fit_items)
            copy_implementation.fit_queries = convert2spark(
                self.fit_queries
            )  # TODO: Use list of '_dataframes' instead convert all dfs manually.
            # Is it needed? Not all of dataframes is needed to convert
        self._impl = copy_implementation
        return self

    def to_pandas(self):
        if self.is_pandas:
            return self
        copy_implementation = self._class_map["pandas"](**self._init_args)
        self._assign_realization_type("pandas")
        copy_implementation = self._copy_base_params_to_new_model(copy_implementation)
        if self.is_fitted:
            self.logger.warning("Converting big dataframes from spark to pandas can cause OOM error.")
            copy_implementation.fit_items = convert2pandas(self.fit_items)
            copy_implementation.fit_queries = convert2pandas(self.fit_queries)
        self._impl = copy_implementation
        return self

    def to_polars(self):
        if self.is_polars:
            return self
        copy_implementation = self._class_map["polars"](**self._init_args)
        self._assign_realization_type("polars")
        copy_implementation = self._copy_base_params_to_new_model(copy_implementation)
        if self.is_fitted:
            self.logger.warning("Converting big dataframes from spark to polars can cause OOM error.")
            copy_implementation.fit_items = convert2polars(self.fit_items)
            copy_implementation.fit_queries = convert2polars(self.fit_queries)
        self._impl = copy_implementation
        return self


class NonPersonolizedRecommenderClient(BaseRecommenderClient, ABC):

    def __init__(self, add_cold_items: bool, cold_weight: float):
        super().__init__()
        self._add_cold_items = add_cold_items
        if 0 < cold_weight <= 1:
            self._cold_weight = cold_weight
        else:
            msg = "`cold_weight` value should be in interval (0, 1]"
            raise ValueError(msg)
        
    @property
    def _init_args(self):
        return {
            "add_cold_items": self._add_cold_items,
            "cold_weight": self._cold_weight,
        }

    @property
    def item_popularity(self):
        if hasattr(self._impl, "item_popularity"):  # add the required attribute setting
            return self._impl.item_popularity
        elif "item_popularity" in self._get_all_attributes_or_functions():
            msg = "Attribute 'item_popularity' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'item_popularity' attribute"
            raise AttributeError(msg)
    
    @property
    def add_cold_items(self):
        if self._impl is not None and hasattr(self._impl, "add_cold_items"):
            return self._impl.add_cold_items
        elif "add_cold_items" in self._get_all_attributes_or_functions():
            msg = "Attribute 'add_cold_items' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'add_cold_items' attribute"
            raise AttributeError(msg)
    
    @add_cold_items.setter
    def add_cold_items(self, value: bool):
        if not isinstance(value, bool) :
            msg = f"incorrect type of argument 'value' ({type(value)}). Use bool"
            raise ValueError(msg)

        self._add_cold_items = value
        if self._impl is not None:
            self._impl.add_cold_items =  self._add_cold_items
        
        

    @property
    def cold_weight(self):
        if self._impl is not None and hasattr(self._impl, "cold_weight"):
            return self._impl.cold_weight
        elif "cold_weight" in self._get_all_attributes_or_functions():
            msg = "Attribute 'cold_weight' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'cold_weight' attribute"
            raise AttributeError(msg)
    
    @add_cold_items.setter
    def cold_weight(self, value: float):
        if not isinstance(value, float) or value == 1:
            msg = f"incorrect type of argument 'value' ({type(value)}). Use float"
            raise ValueError(msg)
        if 0 < value <= 1:
            self._cold_weight = value
            if self._impl is not None:
                self._impl.cold_weight = value
        else:
            msg = "`cold_weight` value should be in interval (0, 1]"
            raise ValueError(msg)


    @item_popularity.setter
    def item_popularity(self, value):
        value_type = (
            "spark"
            if type(value) == SparkDataFrame
            else "pandas" if type(value) == PandasDataFrame else "polars" if type(value) == PolarsDataFrame else None
        )
        if not self._get_realization_type == value_type:
            raise DataModelMissmatchError
        self._impl.item_popularity = value

    @property
    def fill(self):
        return self._impl.fill

    @fill.setter
    def fill(self, value):
        self._impl.fill = value

    @property
    def sample(self):
        if hasattr(self._impl, "sample"):
            return self._impl.sample
        elif "sample" in self._get_all_attributes_or_functions():
            msg = "Attribute 'sample' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'sample' attribute"
            raise AttributeError(msg)

    @property
    def seed(self):
        if hasattr(self._impl, "seed"):
            return self._impl.seed
        elif "seed" in self._get_all_attributes_or_functions():
            msg = "Attribute 'seed' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'seed' attribute"
            raise AttributeError(msg)

    @property
    def _dataframes(self):
        if hasattr(self._impl, "_dataframes"):
            return self._impl._dataframes
        elif "_dataframes" in self._get_all_attributes_or_functions():
            msg = "Attribute '_dataframes' has not been set yet. Set it"
            raise AttributeError(msg)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_dataframes' attribute"
            raise AttributeError(msg)

    def get_items_pd(self, items: SparkDataFrame) -> pd.DataFrame:
        """Clear the cache in spark realization"""
        if hasattr(self._impl, "get_pandas_pd") and isinstance(self._impl, self._class_map["spark"]):
            return self._impl.get_pandas_pd(items)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'get_pandas_pd' function "
            raise AttributeError(msg)
