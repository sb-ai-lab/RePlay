import warnings
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning)

import pandas as pd
import numpy as np
import logging

from absl import app
from absl import flags
from ml_collections import config_flags

from pyspark.sql import SparkSession
from replay.utils.session_handler import get_spark_session, State
from replay.experimental.utils.logger import get_logger
from replay.models import UCB, Wilson, RandomRec

# from replay.experimental.models import LightFMWrap

from replay.experimental.scenarios.obp_wrapper.replay_offline import OBPOfflinePolicyLearner
from replay.experimental.scenarios.obp_wrapper.utils import get_est_rewards_by_reg, bandit_subset

from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.linear_model import LogisticRegression

import obp
from obp.dataset import SyntheticBanditDataset, OpenBanditDataset
from obp.policy import IPWLearner, Random
from obp.ope import OffPolicyEvaluation, DirectMethod, InverseProbabilityWeighting, DoublyRobust


def eval_baselines(dataset, bandit_feedback_train, bandit_feedback_test):
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

    policy_names = ["IPW Learner with Logistic Regression", "IPW Learner with Random Forest", "Unifrom Random"]
    action_dist_list = [action_dist_ipw_lr, action_dist_ipw_rf, action_dist_random]

    ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback_test, ope_estimators=[InverseProbabilityWeighting()])

    for name, action_dist in zip(policy_names, action_dist_list):
        estimated_policy_value = ope.estimate_policy_values(
            action_dist=action_dist,
        )

        print(f"policy value of {name}: {estimated_policy_value}")


def main(_):
    args = FLAGS.config

    spark = State(get_spark_session(**args.spark_params)).session
    spark.sparkContext.setLogLevel("ERROR")

    logger = get_logger("replay", logging.INFO)

    dataset = OpenBanditDataset(behavior_policy=args.behavior_policy, data_path=args.data_path, campaign="all")
    bandit_feedback_train, bandit_feedback_test = dataset.obtain_batch_bandit_feedback(
        test_size=args.test_size, is_timeseries_split=True
    )

    on_policy_policy_value_random = OpenBanditDataset.calc_on_policy_policy_value_estimate(
        behavior_policy="random",
        campaign="all",
        data_path=args.data_path,
        test_size=args.test_size,
        is_timeseries_split=True,
    )

    on_policy_policy_value_bts = OpenBanditDataset.calc_on_policy_policy_value_estimate(
        behavior_policy="bts",
        campaign="all",
        data_path=args.data_path,
        test_size=args.test_size,
        is_timeseries_split=True,
    )

    # eval_baselines(dataset, bandit_feedback_train, bandit_feedback_test)

    model = globals()[args.model](**args.params)

    learner = OBPOfflinePolicyLearner(
        n_actions=dataset.n_actions,
        replay_model=model,
        len_list=dataset.len_list,
    )

    if args.opt.do_opt:
        opt_params = args.opt_params
        bandit_feedback_subset = bandit_subset(opt_params.subset_borders, bandit_feedback_train)
        logger.info(
            learner.optimize(
                bandit_feedback_subset,
                opt_params.val_size,
                param_borders=args.opt.param_borders,
                budget=opt_params.budget,
            )
        )

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
        out_str += f" {key} : {(1e3 * val):.3f},"

    out_str = out_str[:-1]

    logger.info(out_str)
    logger.info("Estimated confidence intervals:")
    logger.info(pd.DataFrame(estimated_ci).to_string())
    logger.info(f"random policy value: {(1e3 * on_policy_policy_value_random):.3f}")
    logger.info(f"bts policy value: {(1e3 * on_policy_policy_value_bts):.3f}")


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config")

    app.run(main)
