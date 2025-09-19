import warnings
from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import NoReturn, Optional, Union

from typing_extensions import TypeAlias

from replay.data import Dataset
from replay.metrics import NDCG, Metric
from replay.models.common import RecommenderCommons
from replay.models.optimization.optuna_objective import ObjectiveWrapper, SplitData, scenario_objective_calculator
from replay.utils import OPTUNA_AVAILABLE, FeatureUnavailableError, FeatureUnavailableWarning

MainObjective = partial(ObjectiveWrapper, objective_calculator=scenario_objective_calculator)

if OPTUNA_AVAILABLE:

    class OptunaMixin(RecommenderCommons):
        """
        A mixin class enabling hyperparameter optimization in a recommender using Optuna objectives.
        """

        _objective = MainObjective
        _search_space: Optional[dict[str, Union[str, Sequence[Union[str, int, float]]]]] = None
        study = None
        criterion: Optional[Metric] = None

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
                    dataset.interactions.select(dataset.feature_schema.query_id_column).distinct(),
                    on=dataset.feature_schema.query_id_column,
                )
            if dataset.item_features is not None:
                item_features = dataset.item_features.join(
                    dataset.interactions.select(dataset.feature_schema.item_id_column).distinct(),
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

        def _check_borders(self, param, borders):
            """Raise value error if param borders are not valid"""
            if param not in self._search_space:
                msg = f"Hyper parameter {param} is not defined for {self!s}"
                raise ValueError(msg)
            if not isinstance(borders, list):
                msg = f"Parameter {param} borders are not a list"
                raise ValueError()
            if self._search_space[param]["type"] != "categorical" and len(borders) != 2:
                msg = f"Hyper parameter {param} is numerical but bounds are not in ([lower, upper]) format"
                raise ValueError(msg)

        def _prepare_param_borders(self, param_borders: Optional[dict[str, list]] = None) -> dict[str, dict[str, list]]:
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
            missing_borders = {param: args[param] for param in search_space if param not in param_borders}
            for param, value in missing_borders.items():
                if search_space[param]["type"] == "categorical":
                    search_space[param]["args"] = [value]
                else:
                    search_space[param]["args"] = [value, value]

            return search_space

        def _init_params_in_search_space(self, search_space):
            """Check if model params are inside search space"""
            params = self._init_args
            outside_search_space = {}
            for param, value in params.items():
                if param not in search_space:
                    continue
                borders = search_space[param]["args"]
                param_type = search_space[param]["type"]

                extra_category = param_type == "categorical" and value not in borders
                param_out_of_bounds = param_type != "categorical" and (value < borders[0] or value > borders[1])
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

        def _params_tried(self):
            """check if current parameters were already evaluated"""
            if self.study is None:
                return False

            params = {name: value for name, value in self._init_args.items() if name in self._search_space}
            return any(params == trial.params for trial in self.study.trials)

        def optimize(
            self,
            train_dataset: Dataset,
            test_dataset: Dataset,
            param_borders: Optional[dict[str, list]] = None,
            criterion: Metric = NDCG,
            k: int = 10,
            budget: int = 10,
            new_study: bool = True,
        ) -> Optional[dict]:
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
            from optuna import create_study
            from optuna.samplers import TPESampler

            self.query_column = train_dataset.feature_schema.query_id_column
            self.item_column = train_dataset.feature_schema.item_id_column
            self.rating_column = train_dataset.feature_schema.interactions_rating_column
            self.timestamp_column = train_dataset.feature_schema.interactions_timestamp_column

            self.criterion = criterion(
                topk=k,
                query_column=self.query_column,
                item_column=self.item_column,
                rating_column=self.rating_column,
            )

            if self._search_space is None:
                self.logger.warning("%s has no hyper parameters to optimize", str(self))
                return None

            if self.study is None or new_study:
                self.study = create_study(direction="maximize", sampler=TPESampler())

            search_space = self._prepare_param_borders(param_borders)
            if self._init_params_in_search_space(search_space) and not self._params_tried():
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

else:
    feature_warning = FeatureUnavailableWarning(
        "Optimization feature not enabled - `optuna` package not found. "
        "Ensure you have the package installed if you want to "
        "use the `optimize()` method in your recommenders."
    )
    warnings.warn(feature_warning)

    class OptunaStub(RecommenderCommons):
        """A stub class to use in case of missing dependencies."""

        def optimize(
            self,
            train_dataset: Dataset,  # noqa: ARG002
            test_dataset: Dataset,  # noqa: ARG002
            param_borders: Optional[dict[str, list]] = None,  # noqa: ARG002
            criterion: Metric = NDCG,  # noqa: ARG002
            k: int = 10,  # noqa: ARG002
            budget: int = 10,  # noqa: ARG002
            new_study: bool = True,  # noqa: ARG002
        ) -> NoReturn:
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
            import sys

            err = FeatureUnavailableError('Cannot use method "optimize()" - Optuna not found.')
            if sys.version_info >= (3, 11):  # pragma: py-lt-311
                err.add_note('To enable this functionality, ensure you have the "optuna" package isntalled.')

            raise err


IsOptimizible: TypeAlias = OptunaMixin if OPTUNA_AVAILABLE else OptunaStub
