from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

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
    def _make_property(attribute_name, has_setter=None):
        def getter(self):
            if self.is_fitted:
                return getattr(self._impl, attribute_name)

        def setter(self, value):
            """Column of fitted items in model"""
            expected_class = (
                DataFrameLike
                if attribute_name.startswith("fit_")
                else str if attribute_name.endwith("_column") else int
            )
            if self.is_fitted and isinstance(value, expected_class):
                setattr(self._impl, attribute_name, value)
            elif not self.is_fitted and isinstance(value, expected_class) and attribute_name[0] != "_":
                self._init_when_first_impl_arrived_args.update({attribute_name: value})
            else:
                msg = f"Can't set to 'fit_items' value {value} in class '{self._impl.__class__}'"
                raise AttributeError(msg)

        if has_setter is not None and attribute_name in has_setter:
            return property(getter, setter)
        else:
            return property(getter)

    attributes_after_fit_with_setter = attributes_after_fit[:2]  # fit_items and fit_queries
    for attr in attributes_after_fit:
        locals()[attr] = _make_property(attr, attributes_after_fit_with_setter)

    def __init__(self):
        self.is_pandas = False
        self.is_spark = False
        self.is_polars = False
        self._init_when_first_impl_arrived_args = {
            "can_predict_cold_queries": False,
            "can_predict_cold_items": False,
            "_search_space": None,
            "_objective": MainObjective,
            "criterion": None,
            "study": None,
        }

    @property
    def is_fitted(self):
        return self._impl is not None

    @property
    @abstractmethod
    def _class_map(self):
        """Map of all implementations (Usually - Spark, Polars and Pandas)"""

    def _assign_implementation_type(self, type_of_model: Literal["pandas", "spark", "polars"]):
        if type_of_model not in ["pandas", "spark", "polars"]:
            msg = f"Argument type_of_model can be spark|pandas|polars, not {type_of_model}"
            raise ValueError(msg)
        self.is_pandas = type_of_model == "pandas"
        self.is_spark = type_of_model == "spark"
        self.is_polars = type_of_model == "polars"

    def _get_implementation_type(self) -> Optional[str]:
        """
        :returns: Stored dataframe type.
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
    def _impl(self):
        """Implementation of model on Spark, Polars or Pandas"""

    @_impl.setter
    @abstractmethod
    def _impl(self, value):
        """Setter of implementation of model on Spark, Polars or Pandas"""

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
        if hasattr(self._impl, "cached_dfs") and self._get_implementation_type() == "spark":
            return self._impl.cached_dfs
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'cached_dfs' attribute"
            raise AttributeError(msg)

    @property
    def fit_items(self):
        """Column of fitted items in model"""
        if hasattr(self._impl, "fit_items"):
            return self._impl.fit_items
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'fit_items' attribute"
            raise AttributeError(msg)

    @fit_items.setter
    def fit_items(self, value):
        """Column of fitted items in model"""
        if self.is_fitter and isinstance(value, DataFrameLike):
            self._impl.fit_items = value
        elif not self.is_fitted and isinstance(value, DataFrameLike):
            self._init_when_first_impl_arrived_args.update({"fit_items": value})
        else:
            msg = f"Can't set to 'fit_items' value {value} in class '{self._impl.__class__}'"
            raise AttributeError(msg)

    @property
    def fit_queries(self):
        """Column of fitted queries in model. Usually, it's column of user_ids"""
        if hasattr(self._impl, "fit_queries"):  # add the required attribute setting
            return self._impl.fit_queries
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'fit_queries' attribute"
            raise AttributeError(msg)

    @fit_queries.setter
    def fit_queries(self, value):
        """Column of fitted queries in model"""
        if self.is_fitted and isinstance(value, DataFrameLike):
            self._impl.fit_queries = value
        elif not self.is_fitted and isinstance(value, DataFrameLike):
            self._init_when_first_impl_arrived_args.update({"fit_queries": value})
        else:
            msg = f"Can't set to 'fit_queries' value ** {value} ** in class '{self._impl.__class__}'"
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
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'model' attribute"
            raise AttributeError(msg)

    @model.setter
    def model(self, value):
        """Column of fitted queries in model"""
        if self.is_fitted:
            self._impl.model = value
        elif not self.is_fitted:
            self._init_when_first_impl_arrived_args.update({"model": value})

    @property
    def can_predict_cold_queries(self):
        if hasattr(self._impl, "can_predict_cold_queries"):
            return self._impl.can_predict_cold_queries
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'can_predict_cold_queries' attribute"
            raise AttributeError(msg)

    @property
    def can_predict_cold_items(self):
        if hasattr(self._impl, "can_predict_cold_items"):
            return self._impl.can_predict_cold_items
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'can_predict_cold_items' attribute"
            raise AttributeError(msg)

    @property
    def _search_space(self):
        if hasattr(self._impl, "_search_space"):
            return self._impl._search_space
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_search_space' attribute"
            raise AttributeError(msg)

    @property
    def _objective(self):
        if hasattr(self._impl, "_objective"):
            return self._impl._objective
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_objective' attribute"
            raise AttributeError(msg)

    @property
    def study(self):
        if hasattr(self._impl, "study"):
            return self._impl.study
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'study' attribute"
            raise AttributeError(msg)

    @study.setter
    def study(self, value):
        if self.is_fitted:
            self._impl.study = value
        else:
            self._init_when_first_impl_arrived_args.update({"study": value})

    @property
    def criterion(self):
        if hasattr(self._impl, "criterion"):
            return self._impl.criterion
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'criterion' attribute"
            raise AttributeError(msg)

    @criterion.setter
    def criterion(self, value):
        if self.is_fitted:
            self._impl.criterion = value
        else:
            self._init_when_first_impl_arrived_args.update({"criterion": value})

    def __str__(self):
        return type(self).__name__

    def _save_model(self, path: str) -> None:
        """
        Method for dump model attributes to disk
        """
        self._impl._save_model(path)

    def _load_model(self, path: str) -> None:
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
        if hasattr(self._impl, "set_params"):
            self._impl.set_params(**params)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'set_params()' function "
            raise AttributeError(msg)

    def _clear_cache(self):
        """Clear the cache in spark realization"""
        if hasattr(self._impl, "_clear_cache") and self._get_implementation_type() == "spark":
            return self._impl._clear_cache
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'cached_dfs' attribute"
            raise AttributeError(msg)

    def _check_input_for_predict_is_correct(self, dataset, queries, items):
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
        if self.is_fitted and hasattr(self._impl, "optimize"):
            return self._impl.optimize(train_dataset, test_dataset, param_borders, criterion, k, budget, new_study)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'optimize()' function "
            raise AttributeError(msg)

    def fit(self, dataset):
        """_RecommenderCommonsSparkImpl._init_args"""
        realization = (
            "spark" if dataset.is_spark else "pandas" if dataset.is_pandas else "polars" if dataset.is_polars else None
        )
        if (
            dataset.is_spark or dataset.is_pandas or dataset.is_polars
        ):  # сначала записать в переменную, затем в self._impl
            new_impl = self._class_map[realization](**self._init_args)
            new_impl.fit(dataset)
            self._impl = new_impl
            self._assign_implementation_type(realization)
            if not self.is_fitted:
                self._impl.set_params(**self._init_when_first_impl_arrived_args)
        else:
            msg = "Model Implementation can't calculate input data due to missmatch of types"
            raise DataModelMissmatchError(msg)

    def fit_predict(
        self,
        dataset: Dataset,
        k: int,
        queries: Optional[Union[DataFrameLike, Iterable]] = None,
        items: Optional[Union[DataFrameLike, Iterable]] = None,
        filter_seen_items: bool = True,
        recs_file_path: Optional[str] = None,
    ) -> Optional[DataFrameLike]:
        """_RecommenderCommonsSparkImpl._init_args"""
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
        """_RecommenderCommonsSparkImpl._init_args"""
        if not self.is_fitted:
            raise NotFittedModelError()
        if dataset is None or dataset.interactions is None:
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
        self, dataset: Dataset, k: int, queries: DataFrameLike, items: DataFrameLike, filter_seen_items: bool = True
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
        self._check_input_for_predict_is_correct(dataset, queries, items)

        recs = self._impl._predict_proba(dataset, k, queries, items, filter_seen_items)
        return recs

    def get_features(
        self, ids: DataFrameLike, features: Optional[DataFrameLike] = None
    ) -> Optional[Tuple[DataFrameLike, int]]:
        """
        Returns query or item feature vectors as a Column with type ArrayType

        :param ids: Spark DataFrame with unique ids
        :return: feature vectors.
            If a model does not have a vector for some ids they are not present in the final result.
        """
        if features is not None:
            # Some of _impl.get_features() have 1 mandatory arg, some have 2 mandatory args
            return self._impl.get_features(ids)
        return self._impl.get_features(ids, features)

    def to_spark(self):
        if not self.is_fitted:
            msg = "Can't convert not fitted model"
            raise NotFittedModelError(msg)
        if self.is_spark:
            return self
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
        copy_implementation._init_when_first_impl_arrived_args = deepcopy(self._init_when_first_impl_arrived_args)
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
        self._impl = copy_implementation
        self._assign_implementation_type("spark")
        return self

    def to_pandas(self):
        if not self.is_fitted:
            msg = "Can't convert not fitted model"
            raise NotFittedModelError(msg)
        if self.is_pandas:
            return self
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
        copy_implementation._init_when_first_impl_arrived_args = deepcopy(self._init_when_first_impl_arrived_args)
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
        self._impl = copy_implementation
        self._assign_implementation_type("pandas")
        return self

    def to_polars(self):
        if not self.is_fitted:
            msg = "Can't convert not fitted model"
            raise NotFittedModelError(msg)
        if self.is_polars:
            return self
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
        copy_implementation._init_when_first_impl_arrived_args = deepcopy(self._init_when_first_impl_arrived_args)
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
        self._impl = copy_implementation
        self._assign_implementation_type("polars")
        return self


class NonPersonolizedRecommenderClient(BaseRecommenderClient, ABC):
    def __init__(self, add_cold_items: bool, cold_weight: float):
        super().__init__()
        self._init_when_first_impl_arrived_args.update(
            {"can_predict_cold_items": True, "can_predict_cold_queries": True, "seed": None}
        )
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
    def add_cold_items(self):
        if self.is_fitted and hasattr(self._impl, "add_cold_items"):
            return self._impl.add_cold_items
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'add_cold_items' attribute"
            raise AttributeError(msg)

    @add_cold_items.setter
    def add_cold_items(self, value: bool):
        if not isinstance(value, bool):
            msg = f"incorrect type of argument 'value' ({type(value)}). Use bool"
            raise ValueError(msg)

        self._add_cold_items = value
        if self.is_fitted:
            self._impl.add_cold_items = self._add_cold_items

    @property
    def cold_weight(self):
        if self.is_fitted and hasattr(self._impl, "cold_weight"):
            return self._impl.cold_weight
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'cold_weight' attribute"
            raise AttributeError(msg)

    @cold_weight.setter
    def cold_weight(self, value: float):
        if not isinstance(value, float) or value != 1:
            msg = f"incorrect type of argument 'value' ({type(value)}). Use float"
            raise ValueError(msg)
        if 0 < value <= 1:
            self._cold_weight = value
            if self.is_fitted:
                self._impl.cold_weight = value
        else:
            msg = "`cold_weight` value should be in interval (0, 1]"
            raise ValueError(msg)

    @property
    def item_popularity(self):
        if hasattr(self._impl, "item_popularity"):  # add the required attribute setting
            return self._impl.item_popularity
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'item_popularity' attribute"
            raise AttributeError(msg)

    @item_popularity.setter
    def item_popularity(self, value):
        value_type = (
            "spark"
            if type(value) == SparkDataFrame
            else "pandas" if type(value) == PandasDataFrame else "polars" if type(value) == PolarsDataFrame else None
        )
        if not self._get_implementation_type == value_type:
            raise DataModelMissmatchError
        self._impl.item_popularity = value

    @property
    def fill(self):
        if hasattr(self._impl, "fill"):
            return self._impl.fill
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'fill' attribute"
            raise AttributeError(msg)

    @fill.setter
    def fill(self, value):
        if self.is_fitted:
            self._impl.fill = value
        else:
            msg = f"Can't set to 'fill' value {value} in class '{self._impl.__class__}'"
            raise AttributeError(msg)

    @property
    def sample(self):
        if hasattr(self._impl, "sample"):
            return self._impl.sample
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'sample' attribute"
            raise AttributeError(msg)

    @property
    def seed(self):
        if hasattr(self._impl, "seed"):
            return self._impl.seed
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'seed' attribute"
            raise AttributeError(msg)

    @property
    def _dataframes(self):
        if hasattr(self._impl, "_dataframes"):
            return self._impl._dataframes
        else:
            msg = f"Class '{self._impl.__class__}' does not have the '_dataframes' attribute"
            raise AttributeError(msg)

    def get_items_pd(self, items: DataFrameLike) -> pd.DataFrame:
        """Clear the cache in spark realization"""
        if hasattr(self._impl, "get_items_pd") and not isinstance(self._impl, self._class_map["pandas"]):
            return self._impl.get_items_pd(items)
        else:
            msg = f"Class '{self._impl.__class__}' does not have the 'get_items_pd' function "
            raise AttributeError(msg)
