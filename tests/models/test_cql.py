# pylint: disable-all
import numpy as np
import pytest
from pyspark.sql import DataFrame

from replay.constants import LOG_SCHEMA
from replay.models import CQL
from tests.utils import spark, log


def test_data_preparation(log: DataFrame):
    """Test MDP dataset preparation is correct."""
    model = CQL(top_k=1, n_epochs=1, action_randomization_scale=1e-9)
    mdp_dataset = model._prepare_mdp_dataset(log)

    # consider only users [0, 1] as the rest has non-deterministic log order
    gt_observations = np.array([
        [0, 0], [0, 2], [0, 1],
        [1, 3], [1, 0],
    ])
    gt_actions = np.array([
        4, 3, 2,
        3, 4
    ])
    gt_rewards = np.array([
        1, 0, 0,
        0, 1,
    ])
    gt_terminals = np.array([
        0, 0, 1,
        0, 1,
    ])
    n = 5
    assert np.array_equal(mdp_dataset.observations[:n], gt_observations)
    # takes into account action randomization noise
    assert np.all(np.abs(mdp_dataset.actions[:n].flatten() - gt_actions) < 1e-2)
    assert np.array_equal(mdp_dataset.rewards[:n], gt_rewards)
    assert np.array_equal(mdp_dataset.terminals[:n], gt_terminals)


def test_works_predict(log: DataFrame):
    """Test fit/predict works, and that the model correctly filters out seen items."""
    model = CQL(top_k=1, n_epochs=1)
    model.fit(log)
    recs = model.predict(log, k=1).toPandas()
    assert recs.loc[recs["user_idx"] == 0, "item_idx"].iloc[0] == 3
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] in [1, 2]
    assert recs.loc[recs["user_idx"] == 2, "item_idx"].iloc[0] == 3


def test_works_predict_pairs(log: DataFrame):
    """Test fit/predict_pairs works, and that the model outputs correct number of items."""
    top_k = 3
    model = CQL(top_k=top_k, n_epochs=1)
    model.fit(log)
    user_item_pairs = log.select("user_idx", "item_idx")
    recs = model.predict_pairs(user_item_pairs, log, k=top_k).toPandas()
    assert np.count_nonzero(recs["user_idx"] == 0) == 3
    assert np.count_nonzero(recs["user_idx"] == 1) == 2
    assert np.count_nonzero(recs["user_idx"] == 2) == 3


def test_serialize_deserialize_policy(log: DataFrame):
    """Test that serializing and deserializing the policy does not change predictions."""
    model = CQL(top_k=1, n_epochs=1)
    model.fit(log)

    items_batch = np.array([
        [0, 0], [0, 2], [0, 1],
        [1, 3], [1, 0],
    ])

    # predict user-item relevance with the original model
    relevance = model.model.predict(items_batch)

    # predict user-item relevance with the serialized-then-restored policy
    restored_policy = model._deserialize_policy(model._serialize_policy())
    restored_relevance = model._predict_relevance_with_policy(restored_policy, items_batch)
    assert relevance == pytest.approx(restored_relevance, abs=1e-5)
