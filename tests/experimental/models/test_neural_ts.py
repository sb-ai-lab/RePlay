# pylint: disable-all
import os
from datetime import datetime

import pytest

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

import numpy as np
from obp.dataset import OpenBanditDataset
from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting, DirectMethod, DoublyRobust
from pyspark.sql import functions as sf

from replay.data import get_schema
from replay.experimental.models import NeuralTS
from replay.experimental.scenarios.obp_wrapper.replay_offline import OBPOfflinePolicyLearner
from replay.experimental.scenarios.obp_wrapper.utils import get_est_rewards_by_reg
from tests.utils import sparkDataFrameEqual

SEED = 123


@pytest.fixture
def log(spark):
    date1 = datetime(2019, 1, 1)
    date2 = datetime(2019, 1, 2)
    date3 = datetime(2019, 1, 3)
    return spark.createDataFrame(
        data=[
            [0, 0, date1, 1.0],
            [1, 0, date1, 1.0],
            [2, 1, date2, 2.0],
            [2, 1, date2, 2.0],
            [1, 1, date3, 2.0],
            [2, 2, date3, 2.0],
            [0, 2, date3, 2.0],
        ],
        schema=get_schema("user_idx", "item_idx", "timestamp", "relevance"),
    )


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame([(0, 2.0, 5.0), (1, 0.0, -5.0), (2, 4.0, 3.0)]).toDF(
        "user_idx", "user_feature_cat", "user_feature_cont"
    )


@pytest.fixture
def item_features(spark):
    return spark.createDataFrame([(0, 4.0, 5.0), (1, 5.0, 4.0), (2, 1.5, 3.3)]).toDF(
        "item_idx", "item_feature_cat", "item_feature_cont"
    )


@pytest.fixture
def model():
    model = NeuralTS(
        embedding_sizes=[2, 2, 4],
        hidden_layers=[2, 5],
        wide_out_dim=1,
        deep_out_dim=1,
        dim_head=1,
        n_epochs=1,
        use_gpu=False,
        cnt_neg_samples=1,
        cnt_samples_for_predict=2,
    )

    return model


@pytest.fixture
def model_with_features():
    cols_item = {
        "continuous_cols": ["item_feature_cont"],
        "cat_embed_cols": ["item_feature_cat"],
        "wide_cols": ["item_feature_cat", "item_feature_cont"],
    }

    cols_user = {
        "continuous_cols": ["user_feature_cont"],
        "cat_embed_cols": ["user_feature_cat"],
        "wide_cols": ["user_feature_cat", "user_feature_cont"],
    }

    model_with_features = NeuralTS(
        user_cols=cols_user,
        item_cols=cols_item,
        embedding_sizes=[2, 2, 4],
        hidden_layers=[2, 5],
        wide_out_dim=1,
        deep_out_dim=1,
        dim_head=1,
        n_epochs=1,
        use_gpu=False,
        cnt_neg_samples=1,
        cnt_samples_for_predict=2,
    )

    return model_with_features


@pytest.mark.experimental
def test_equal_preds(model, user_features, item_features, log):
    dir_name = "test"
    model.fit(log, user_features=user_features, item_features=item_features)
    torch.manual_seed(SEED)
    base_pred = model.predict(log, 5, user_features=user_features, item_features=item_features)
    model.model_save(dir_name)
    model.model_load(dir_name)
    torch.manual_seed(SEED)
    new_pred = model.predict(log, 5, user_features=user_features, item_features=item_features)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.experimental
def test_use_features(model_with_features, user_features, item_features, log):
    model_with_features.fit(log, user_features=user_features, item_features=item_features)

    users = user_features.select("user_idx").distinct()
    items = item_features.select("item_idx").distinct()

    base_pred = model_with_features._predict(
        log, 5, users=users, items=items, user_features=user_features, item_features=item_features
    )

    n_users = users.count()
    n_items = items.count()

    assert base_pred.count() == n_users * n_items


@pytest.mark.experimental
def test_predict_pairs(log, user_features, item_features, model):
    model.fit(log, user_features, item_features)

    pred = model.predict_pairs(
        log.filter(sf.col("user_idx") == 1).select("user_idx", "item_idx"),
        user_features=user_features,
        item_features=item_features,
    )
    assert pred.count() == 2
    assert pred.select("user_idx").distinct().collect()[0][0] == 1


@pytest.mark.experimental
def test_predict_empty_log(model_with_features, user_features, item_features, log):
    model_with_features.fit(log, user_features=user_features, item_features=item_features)
    users = user_features.select("user_idx").distinct()
    items = item_features.select("item_idx").distinct()

    model_with_features._predict(
        log.limit(0), 1, users=users, items=items, user_features=user_features, item_features=item_features
    )

@pytest.fixture
def obp_dataset():
    dataset = OpenBanditDataset(behavior_policy="random", data_path=None, campaign="all")
    return dataset

@pytest.fixture
def obp_learner(obp_dataset):
    obp_model = NeuralTS(
        embedding_sizes=[2, 2, 4],
        hidden_layers=[2, 5],
        wide_out_dim=1,
        deep_out_dim=1,
        dim_head=1,
        n_epochs=1,
        use_gpu=False,
        cnt_neg_samples=1,
        cnt_samples_for_predict=2,
        cnt_users=obp_dataset.n_rounds,
    )
    learner = OBPOfflinePolicyLearner(
        n_actions=obp_dataset.n_actions,
        replay_model=obp_model,
        len_list=obp_dataset.len_list,
    )
    return learner

def test_obp_evaluation(obp_dataset, obp_learner):
    bandit_feedback_train, _ = obp_dataset.obtain_batch_bandit_feedback(
        test_size=0.3, is_timeseries_split=True
    )
    _, bandit_feedback_test = obp_dataset.obtain_batch_bandit_feedback(
        test_size=0.3, is_timeseries_split=True
    )

    obp_learner.fit(
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        timestamp=np.arange(bandit_feedback_train["n_rounds"]),
        context=bandit_feedback_train["context"],
        action_context=bandit_feedback_train["action_context"],
    )
    action_dist = obp_learner.predict(
        bandit_feedback_test["n_rounds"],
        bandit_feedback_test["context"],
    )
    assert action_dist.shape == (bandit_feedback_test["n_rounds"], obp_learner.n_actions, obp_learner.len_list)

    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback_test,
        ope_estimators=[InverseProbabilityWeighting(), DirectMethod(), DoublyRobust()],
    )
    estimated_rewards_by_reg_model = get_est_rewards_by_reg(
        obp_dataset.n_actions, obp_dataset.len_list, bandit_feedback_test, bandit_feedback_test
    )
    estimated_policy_value = ope.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    assert "ipw" in estimated_policy_value
    assert "dm" in estimated_policy_value
    assert "dr" in estimated_policy_value
