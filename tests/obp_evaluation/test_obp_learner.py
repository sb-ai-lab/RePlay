import pytest
import numpy as np
import pandas as pd

from replay.obp_evaluation.replay_offline import RePlayOfflinePolicyLearner
from replay.obp_evaluation.utils import split_bandit_feedback
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
        "position": None
    }


@pytest.fixture
def bandit_log():
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


@pytest.fixture
def replay_obp_learner(model, bandit_feedback):
    learner = RePlayOfflinePolicyLearner(n_actions=2,
                                         replay_model=model,
                                         len_list=1)

    assert type(learner.replay_model) == RandomRec

    learner.fit(action=bandit_feedback["action"],
                reward=bandit_feedback["reward"],
                timestamp=bandit_feedback["timestamp"],
                context=bandit_feedback["context"],
                action_context=bandit_feedback["action_context"])

    return learner


def test_fit(model, bandit_feedback, replay_obp_learner):
    assert replay_obp_learner.max_usr_id == 3

    train, val = split_bandit_feedback(bandit_feedback, val_size=0.3)

    assert train["n_rounds"] == 2
    assert val["n_rounds"] == 1


@pytest.mark.parametrize("context", [[[1, 1, 1]], [[0.5, 1, 1]]])
def test_predict(context, replay_obp_learner):
    pred = replay_obp_learner.predict(1, np.array(context, dtype=np.float32))

    assert replay_obp_learner.max_usr_id == 4


@pytest.mark.parametrize("val_size", [0.3])
def test_optimize(bandit_feedback, replay_obp_learner, val_size):
    best_params = replay_obp_learner.optimize(bandit_feedback, val_size, budget=2)

    assert replay_obp_learner.replay_model.alpha == best_params["alpha"]
    assert replay_obp_learner.replay_model.distribution == best_params["distribution"]
