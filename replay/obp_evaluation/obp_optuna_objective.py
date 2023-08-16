import numpy as np
import pandas as pd
import collections
from functools import partial
from typing import Any, Dict, List, Optional, Callable, Union

from optuna import Trial
from replay.optuna_objective import ObjectiveWrapper

from obp.ope import (
    OffPolicyEvaluation,
    DirectMethod,
    InverseProbabilityWeighting,
    DoublyRobust
)
from replay.obp_evaluation.utils import get_est_rewards_by_reg
from replay.optuna_objective import suggest_params

def obp_objective_calculator(
        trial: Trial,
        search_space: Dict[str, List[Optional[Any]]],
        bandit_feedback_train: Dict[str, np.ndarray],
        bandit_feedback_val: Dict[str, np.ndarray],
        learner,
        criterion: str,
        k: int,
    ) -> float:
    """
    Sample parameters and calculate criterion value
    :param trial: optuna trial
    :param search_space: hyper parameter search space
    :bandit_feedback_train: dict with bandit train data
    :bandit_feedback_cal: dist with bandit validation data
    :param criterion: optimization metric
    :param k: length of a recommendation list
    :return: criterion value
    """

    params_for_trial = suggest_params(trial, search_space)
    learner.replay_model.set_params(**params_for_trial)

    timestamp = None
    if "timestamp" in bandit_feedback_train.keys():
        timestamp = bandit_feedback_train["timestamp"]
    else:
        timestamp = np.arange(bandit_feedback_train["n_rounds"])

    learner.fit(action=bandit_feedback_train["action"],
                reward=bandit_feedback_train["reward"],
                timestamp=timestamp,
                context=bandit_feedback_train["context"],
                action_context=bandit_feedback_train["action_context"]
            )

    action_dist = learner.predict(bandit_feedback_val["n_rounds"],
                                  bandit_feedback_val["context"]
                                )

    ope_estimator = None
    if criterion == "ipw":
        ope_estimator = InverseProbabilityWeighting()
    elif criterion == "dm":
        ope_estimator = DirectMethod()
    elif criterion == "dr":
        ope_estimator = DoublyRobust()
    else:
        raise NotImplementedError(f"There is no criterion with name {criterion}")

    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback_val,
        ope_estimators=[ope_estimator]
    )

    estimated_rewards_by_reg_model = None
    if criterion == "dm" or criterion == "dr":
        estimated_rewards_by_reg_model = get_est_rewards_by_reg(learner.n_actions,
                                                                k,
                                                                bandit_feedback_train,
                                                                bandit_feedback_val)

    estimated_policy_value = ope.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )[criterion]

    return estimated_policy_value

OBPObjective = partial(
    ObjectiveWrapper, objective_calculator=obp_objective_calculator
)