import logging
import sys

import numpy as np
import pytest

if sys.version_info > (3, 9):
    pytest.skip(
        reason="obp does't support 3.10",
        allow_module_level=True,
    )

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

from replay.experimental.scenarios.obp_wrapper.replay_offline import OBPOfflinePolicyLearner
from replay.experimental.scenarios.obp_wrapper.utils import split_bandit_feedback
from replay.experimental.utils.logger import get_logger
from replay.models import RandomRec


@pytest.fixture
def bandit_feedback():
    return {
        "n_rounds": 3,
        "n_actions": 2,
        "action": np.array([1, 0, 1]),
        "reward": np.array([1, 0, 0]),
        "context": np.array([[0.5, 1, 1], [1, 0.5, 1], [1, 1, 0.5]]),
        "action_context": np.array([[2, 0.5, 0.5], [0.5, 2, 0.5], [0.5, 0.5, 2]]),
        "timestamp": np.array([1, 2, 3]),
        "pscore": np.array([0.5, 0.5, 0.5]),
        "position": None,
    }


@pytest.fixture
@pytest.mark.usefixtures("spark", "LOG_SCHEMA")
def bandit_log(spark, LOG_SCHEMA):
    return spark.createDataFrame(
        data=[
            [1, 1, 1, 1.0],
            [2, 0, 2, 0.0],
            [3, 1, 3, 0.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def model():
    return RandomRec(seed=42)


def test_logger():
    _ = get_logger("replay", logging.INFO)


@pytest.fixture
def replay_obp_learner(model, bandit_feedback):
    learner = OBPOfflinePolicyLearner(n_actions=2, replay_model=model, len_list=1)

    assert type(learner.replay_model) is RandomRec

    learner.fit(
        action=bandit_feedback["action"],
        reward=bandit_feedback["reward"],
        timestamp=bandit_feedback["timestamp"],
        context=bandit_feedback["context"],
        action_context=bandit_feedback["action_context"],
    )

    _ = learner.logger

    return learner


def test_fit(model, bandit_feedback, replay_obp_learner):
    n_rounds = bandit_feedback["n_rounds"]

    assert replay_obp_learner.max_usr_id == n_rounds

    train, val = split_bandit_feedback(bandit_feedback, val_size=0.3)

    n_rounds_train = int(0.7 * bandit_feedback["n_rounds"])
    assert train["n_rounds"] == n_rounds_train
    assert val["n_rounds"] == n_rounds - n_rounds_train


@pytest.mark.parametrize("context", [[[1, 1, 1]], [[0.5, 1, 1]]])
def test_predict(context, replay_obp_learner):
    n_rounds = 1
    pred = replay_obp_learner.predict(n_rounds, np.array(context, dtype=np.float32))

    assert replay_obp_learner.max_usr_id == len(context[0]) + n_rounds
    assert pred.shape == (n_rounds, replay_obp_learner.n_actions, replay_obp_learner.len_list)

    assert np.allclose(pred.sum(1), np.ones(shape=(n_rounds, replay_obp_learner.len_list)))


@pytest.mark.parametrize("val_size,criterion", [(0.3, "ipw"), (0.3, "dm"), (0.3, "dr")])
def test_optimize(bandit_feedback, replay_obp_learner, val_size, criterion):
    best_params = replay_obp_learner.optimize(bandit_feedback, val_size, budget=2, criterion=criterion)

    assert replay_obp_learner.replay_model.alpha == best_params["alpha"]
    assert replay_obp_learner.replay_model.distribution == best_params["distribution"]
