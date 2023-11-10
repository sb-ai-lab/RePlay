"""
This class calculates loss function for optimization process
"""
import collections
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Callable, Union

from optuna import Trial
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame as SparkDataFrame

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


def suggest_params(
    trial: Trial, search_space: Dict[str, Dict[str, Union[str, List[Any]]]],
) -> Dict[str, Any]:
    """
    This function suggests params to try.

    :param trial: optuna trial
    :param search_space: hyper parameters and their bounds
    :return: dict with parameter values
    """
    suggest_dict = {
        "uniform": trial.suggest_uniform,
        "int": trial.suggest_int,
        "loguniform": trial.suggest_loguniform,
        "loguniform_int": partial(trial.suggest_int, log=True),
    }

    res = {}
    for param in search_space:
        border = search_space[param]["args"]
        param_type = search_space[param]["type"]
        if param_type == "categorical":
            res[param] = trial.suggest_categorical(param, border)
        else:
            low, high = border
            suggest_fn = suggest_dict[param_type]
            res[param] = suggest_fn(param, low=low, high=high)
    return res


def calculate_criterion_value(
    criterion: Metric,
    k: int,
    recommendations: SparkDataFrame,
    ground_truth: SparkDataFrame
) -> float:
    """
    Calculate criterion value for given parameters
    :param criterion: optimization metric
    :param k: length of a recommendation list
    :param recommendations: calculated recommendations
    :param ground_truth: test data
    :return: criterion value
    """
    result_dict = criterion(
        k,
        query_column="user_idx",
        item_column="item_idx",
        rating_column="relevance",
    )(recommendations, ground_truth)
    return list(result_dict.values())[0]


def eval_quality(
    split_data: SplitData, recommender, criterion: Metric, k: int,
) -> float:
    """
    Calculate criterion value using model, data and criterion parameters
    :param split_data: data to train and test model
    :param recommender: recommender model
    :param criterion: optimization metric
    :param k: length of a recommendation list
    :return: criterion value
    """
    logger = logging.getLogger("replay")
    logger.debug("Fitting model inside optimization")
    # pylint: disable=protected-access
    recommender._fit_wrap(
        split_data.train,
        split_data.user_features_train,
        split_data.item_features_train,
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
    criterion_value = calculate_criterion_value(criterion, k, recs, split_data.test)
    logger.debug("%s=%.6f", criterion, criterion_value)
    return criterion_value


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
    Sample parameters and calculate criterion value
    :param trial: optuna trial
    :param search_space: hyper parameter search space
    :param split_data: data to train and test model
    :param recommender: recommender model
    :param criterion: optimization metric
    :param k: length of a recommendation list
    :return: criterion value
    """
    params_for_trial = suggest_params(trial, search_space)
    recommender.set_params(**params_for_trial)
    return eval_quality(split_data, recommender, criterion, k)


MainObjective = partial(
    ObjectiveWrapper, objective_calculator=scenario_objective_calculator
)


# pylint: disable=too-few-public-methods
class ItemKNNObjective:
    """
    This class is implemented according to
    `instruction <https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments>`_
    on integration with ``optuna``.

    Criterion is calculated with ``__call__``,
    other arguments are passed into ``__init__``.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        max_neighbours = self.kwargs["search_space"]["num_neighbours"]["args"][
            1
        ]
        model = self.kwargs["recommender"]
        split_data = self.kwargs["split_data"]
        train = split_data.train
        model.num_neighbours = max_neighbours

        df = train.select("user_idx", "item_idx", "relevance")
        if not model.use_relevance:
            df = df.withColumn("relevance", sf.lit(1))

        self.dot_products = model._get_products(df).cache()

    def objective_calculator(
        self,
        trial: Trial,
        search_space: Dict[str, List[Optional[Any]]],
        split_data: SplitData,
        recommender,
        criterion: Metric,
        k: int,
    ) -> float:
        """
        Sample parameters and calculate criterion value
        :param trial: optuna trial
        :param search_space: hyper parameter search space
        :param split_data: data to train and test model
        :param recommender: recommender model
        :param criterion: optimization metric
        :param k: length of a recommendation list
        :return: criterion value
        """
        params_for_trial = suggest_params(trial, search_space)
        recommender.set_params(**params_for_trial)
        recommender.fit_users = split_data.train.select("user_idx").distinct()
        recommender.fit_items = split_data.train.select("item_idx").distinct()
        similarity = recommender._shrink(self.dot_products, recommender.shrink)
        recommender.similarity = recommender._get_k_most_similar(
            similarity
        ).cache()
        recs = recommender._predict_wrap(
            log=split_data.train,
            k=k,
            users=split_data.users,
            items=split_data.items,
            user_features=split_data.user_features_test,
            item_features=split_data.item_features_test,
        )
        logger = logging.getLogger("replay")
        logger.debug("Calculating criterion")
        criterion_value = calculate_criterion_value(criterion, k, recs, split_data.test)
        logger.debug("%s=%.6f", criterion, criterion_value)
        return criterion_value

    def __call__(self, trial: Trial) -> float:
        """
        Calculate criterion for ``optuna``.

        :param trial: current trial
        :return: criterion value
        """
        return self.objective_calculator(trial=trial, **self.kwargs)
