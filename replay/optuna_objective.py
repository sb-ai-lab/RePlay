"""
This class calculates loss function for optimization process
"""
import collections
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Callable, Union

from optuna import Trial

from replay.metrics.base_metric import Metric

SplitData = collections.namedtuple(
    "SplitData",
    "train test users items user_features_train "
    "user_features_test item_features_train item_features_test",
)


# pylint: disable=too-few-public-methods
class ObjectiveWrapper:
    """
    This class is implemented according to
    `instruction <https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments>`_
    on integration with ``optuna``.

    Criterion is calculated with ``__call__``,
    other arguments are passed into ``__init__``.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes

    def __init__(
        self, objective_calculator: Callable[..., float], **kwargs: Any
    ):
        self.objective_calculator = objective_calculator
        self.kwargs = kwargs

    def __call__(self, trial: Trial) -> float:
        """
        Calculate criterion for ``optuna``.

        :param trial: current trial
        :return: criterion value
        """
        return self.objective_calculator(trial=trial, **self.kwargs)


def suggest_param_value(
    trial: Trial,
    param_name: str,
    param_bounds: List[Optional[Any]],
    default_params_data: Dict[str, Dict[str, Union[str, List[Any]]]],
) -> Union[str, float, int]:
    """
    This function calls trial method dependent on hyper parameter type provided.

    :param trial: optuna trial
    :param param_name: parameter name
    :param param_bounds: lower and upper search bounds or list of categorical values.
        If list is empty, default values are used.
    :param default_params_data: hyper parameters and their default bounds
    :return: hyper parameter value
    """
    to_optuna_types_dict = {
        "uniform": trial.suggest_uniform,
        "int": trial.suggest_int,
        "loguniform": trial.suggest_loguniform,
        "loguniform_int": partial(trial.suggest_int, log=True),
    }

    if param_name not in default_params_data:
        raise ValueError(
            f"Hyper parameter {param_name} is not defined for this model"
        )
    param_type = default_params_data[param_name]["type"]
    param_args = (
        param_bounds
        if param_bounds
        else default_params_data[param_name]["args"]
    )
    if param_type == "categorical":
        return trial.suggest_categorical(param_name, param_args)

    if len(param_args) != 2:
        raise ValueError(
            f"""
Hyper parameter {param_name} is numerical but no bounds
([lower, upper]) were provided"""
        )
    lower, upper = param_args

    return to_optuna_types_dict[param_type](param_name, low=lower, high=upper)


# pylint: disable=too-many-arguments
def scenario_objective_calculator(
    trial: Trial,
    search_space: Dict[str, List[Optional[Any]]],
    split_data: SplitData,
    recommender,
    criterion: Metric,
    k: int,
) -> float:
    """
    Calculate criterion value for given parameters
    :param trial: optuna trial
    :param search_space: hyper parameter search space
    :param split_data: data to train and test model
    :param recommender: recommender model
    :param criterion: optimization metric
    :param k: length of a recommendation list
    :return: criterion value
    """
    logger = logging.getLogger("replay")

    params_for_trial = {}
    for param_name, param_data in search_space.items():
        params_for_trial[param_name] = suggest_param_value(
            # pylint: disable=protected-access
            trial,
            param_name,
            param_data,
            recommender._search_space,
        )

    recommender.set_params(**params_for_trial)
    logger.debug("Fitting model inside optimization")
    # pylint: disable=protected-access
    recommender._fit_wrap(
        split_data.train,
        split_data.user_features_train,
        split_data.item_features_train,
        False,
    )
    logger.debug("Predicting inside optimization")
    recs = recommender._predict_wrap(
        log=split_data.train,
        k=k,
        users=split_data.users,
        items=split_data.items,
        user_features=split_data.user_features_test,
        item_features=split_data.item_features_test,
    )
    logger.debug("Calculating criterion")
    criterion_value = criterion(recs, split_data.test, k)
    logger.debug("%s=%.6f", criterion, criterion_value)
    return criterion_value


MainObjective = partial(
    ObjectiveWrapper, objective_calculator=scenario_objective_calculator
)
