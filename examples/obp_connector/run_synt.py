import warnings
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning)

import pandas as pd
import numpy as np
import logging

from replay.utils.session_handler import get_spark_session, State
from replay.experimental.utils.logger import get_logger

from replay.models import UCB, Wilson, RandomRec, PopRec
from replay.experimental.scenarios.obp_wrapper.replay_offline import OBPOfflinePolicyLearner
from replay.experimental.scenarios.obp_wrapper.utils import get_est_rewards_by_reg

from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.linear_model import LogisticRegression

import obp
from obp.dataset import SyntheticBanditDataset, logistic_reward_function
from obp.policy import IPWLearner, NNPolicyLearner, Random

from obp.ope import OffPolicyEvaluation, DirectMethod, InverseProbabilityWeighting, DoublyRobust


def eval_baselines(dataset, bandit_feedback_train, bandit_feedback_test):
    func = lambda op_type: NNPolicyLearner(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        off_policy_objective=op_type,
        len_list=dataset.len_list,
        batch_size=64,
        random_state=12345,
    )

    nn_methods = [func(op_obj) for op_obj in ["dm", "ipw", "dr"]]

    for m in nn_methods:
        m.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
            position=bandit_feedback_train["position"],
        )

    action_dist_nn_dm, action_dist_nn_ipw, action_dist_nn_dr = [
        m.predict_proba(context=bandit_feedback_test["context"]) for m in nn_methods
    ]

    clss = [
        LogisticRegression(C=100, random_state=12345),
        RandomForest(n_estimators=30, min_samples_leaf=10, random_state=12345),
    ]

    ipw_methods = [IPWLearner(n_actions=dataset.n_actions, base_classifier=c, len_list=dataset.len_list) for c in clss]

    for m in ipw_methods:
        m.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
            position=bandit_feedback_train["position"],
        )

    action_dist_ipw_lr, action_dist_ipw_rf = [m.predict(context=bandit_feedback_test["context"]) for m in ipw_methods]

    random = Random(n_actions=dataset.n_actions, len_list=dataset.len_list)

    # compute the action choice probabilities for the test set
    action_dist_random = random.compute_batch_action_dist(n_rounds=bandit_feedback_test["n_rounds"])

    policy_names = [
        "NN Policy Learner with DM",
        "NN Policy Learner with IPW",
        "NN Policy Learner with DR",
        "IPW Learner with Logistic Regression",
        "IPW Learner with Random Forest",
        "Unifrom Random",
    ]
    action_dist_list = [
        action_dist_nn_dm,
        action_dist_nn_ipw,
        action_dist_nn_dr,
        action_dist_ipw_lr,
        action_dist_ipw_rf,
        action_dist_random,
    ]

    ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback_test, ope_estimators=[InverseProbabilityWeighting()])

    for name, action_dist in zip(policy_names, action_dist_list):
        estimated_policy_value = ope.estimate_policy_values(
            action_dist=action_dist,
        )

        print(f"policy value of {name}: {estimated_policy_value}")


if __name__ == "__main__":
    spark = State(get_spark_session()).session
    spark.sparkContext.setLogLevel("ERROR")

    logger = get_logger("replay", logging.INFO)

    dataset = SyntheticBanditDataset(
        n_actions=10,
        dim_context=5,
        beta=-2,  # invers e temperature parameter to control the optimality and entropy of the behavior policy
        reward_type="binary",  # "binary" or "continuous"
        reward_function=logistic_reward_function,
        random_state=12345,
    )

    n_rounds_train, n_rounds_test = 10000, 10000
    bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds_train)
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds_test)

    eval_baselines(dataset, bandit_feedback_train, bandit_feedback_test)

    model = UCB(exploration_coef=2.0)

    learner = OBPOfflinePolicyLearner(
        n_actions=dataset.n_actions,
        replay_model=model,
        len_list=dataset.len_list,
    )

    # In case if optimization
    # param_borders = {
    #     "coef": [-5, 5]
    # }
    # logger.info(learner.optimize(bandit_feedback_train, 0.3, param_borders=param_borders, budget=50))

    timestamp = np.arange(bandit_feedback_train["n_rounds"])

    learner.fit(
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        timestamp=timestamp,
        context=bandit_feedback_train["context"],
        action_context=bandit_feedback_train["action_context"],
    )

    action_dist = learner.predict(bandit_feedback_test["n_rounds"], bandit_feedback_test["context"])

    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback_test,
        ope_estimators=[InverseProbabilityWeighting(), DirectMethod(), DoublyRobust()],
    )

    estimated_rewards_by_reg_model = get_est_rewards_by_reg(
        dataset.n_actions, dataset.len_list, bandit_feedback_train, bandit_feedback_test
    )

    estimated_policy_value = ope.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )

    estimated_ci = ope.estimate_intervals(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        n_bootstrap_samples=10000,
        random_state=12345,
    )

    out_str = f"Scores for {model.__class__.__name__}:"
    for key, val in estimated_policy_value.items():
        out_str += f" {key} : {(val):.3f},"

    out_str = out_str[:-1]

    logger.info(out_str)
    logger.info("Estimated confidence intervals:")
    logger.info(pd.DataFrame(estimated_ci).to_string())
