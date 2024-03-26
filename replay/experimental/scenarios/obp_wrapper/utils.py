from typing import Dict, List, Tuple

import numpy as np
from obp.ope import RegressionModel
from sklearn.linear_model import LogisticRegression


def get_est_rewards_by_reg(n_actions, len_list, bandit_feedback_train, bandit_feedback_test):
    """
    Fit Logistic Regression to rewards from `bandit_feedback`.
    """
    regression_model = RegressionModel(
        n_actions=n_actions,
        len_list=len_list,
        action_context=bandit_feedback_train["action_context"],
        base_model=LogisticRegression(max_iter=1000, random_state=12345),
    )

    regression_model.fit(
        context=bandit_feedback_train["context"],
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        position=bandit_feedback_train["position"],
        pscore=bandit_feedback_train["pscore"],
    )

    estimated_rewards_by_reg_model = regression_model.predict(
        context=bandit_feedback_test["context"],
    )

    return estimated_rewards_by_reg_model


def bandit_subset(borders: List[int], bandit_feedback: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    This function returns subset of a `bandit_feedback`
    with borders specified in `borders`.

    :param bandit_feedback: Bandit log data with fields
                            ``[action, reward, context, action_context,
                               n_rounds, n_actions, position, pscore]``
                            as in OpenBanditPipeline.
    :param borders: List with two values ``[left, right]``
    :return: Returns subset of a `bandit_feedback` for each key with
             indexes from `left`(including) to `right`(excluding).
    """
    assert len(borders) == 2

    left, right = borders

    assert left < right

    position = None if bandit_feedback["position"] is None else bandit_feedback["position"][left:right]

    return {
        "n_rounds": right - left,
        "n_actions": bandit_feedback["n_actions"],
        "action": bandit_feedback["action"][left:right],
        "position": position,
        "reward": bandit_feedback["reward"][left:right],
        "pscore": bandit_feedback["pscore"][left:right],
        "context": bandit_feedback["context"][left:right],
        "action_context": bandit_feedback["action_context"][left:right],
    }


def split_bandit_feedback(
    bandit_feedback: Dict[str, np.ndarray], val_size: int = 0.3
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split `bandit_feedback` into two subsets.
    :param bandit_feedback: Bandit log data with fields
                            ``[action, reward, context, action_context,
                               n_rounds, n_actions, position, pscore]``
                            as in OpenBanditPipeline.
    :param val_size: Number in range ``[0, 1]`` corresponding to the proportion of
                     train/val split.
    :return: `bandit_feedback_train` and `bandit_feedback_val` split.
    """

    n_rounds = bandit_feedback["n_rounds"]
    n_rounds_train = int(n_rounds * (1.0 - val_size))

    bandit_feedback_train = bandit_subset([0, n_rounds_train], bandit_feedback)
    bandit_feedback_val = bandit_subset([n_rounds_train, n_rounds], bandit_feedback)

    return bandit_feedback_train, bandit_feedback_val
