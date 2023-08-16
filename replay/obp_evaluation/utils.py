from sklearn.linear_model import LogisticRegression
from obp.ope import RegressionModel
import numpy as np

def get_est_rewards_by_reg(n_actions, len_list, bandit_feedback_train, bandit_feedback_test):
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

def bandit_subset(borders, bandit_feedback):
    l, r = borders

    assert l < r

    position = None if bandit_feedback["position"] is None else bandit_feedback["position"][l:r]

    return dict(
        n_rounds=r - l,
        n_actions=bandit_feedback["n_actions"],
        action=bandit_feedback["action"][l:r],
        position=position,
        reward=bandit_feedback["reward"][l:r],
        pscore=bandit_feedback["pscore"][l:r],
        context=bandit_feedback["context"][l:r],
        action_context=bandit_feedback["action_context"][l:r]
    )

def split_bandit_feedback(bandit_feedback, val_size=0.3):
    '''
        bandit_feedback is a Dict with fields ["action", "reward", "context",
                                               "action_context", "n_rounds",
                                               "n_actions", "position", "pscore"]
    '''

    n_rounds = bandit_feedback["n_rounds"]
    n_rounds_train = np.int32(n_rounds * (1.0 - val_size))

    bandit_feedback_train = bandit_subset([0, n_rounds_train], bandit_feedback)
    bandit_feedback_val = bandit_subset([n_rounds_train, n_rounds], bandit_feedback)

    return bandit_feedback_train, bandit_feedback_val