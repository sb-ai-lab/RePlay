# pylint: disable-all
import numpy as np
from _pytest.python_api import approx
from pytest import approx
from pyspark.sql import DataFrame

from replay.data import LOG_SCHEMA
from replay.models.cql import MdpDatasetBuilder
from replay.models import CQL
from tests.utils import spark, log


def test_predict_filters_out_seen_items(log: DataFrame):
    """Test that fit/predict works, and that the model correctly filters out seen items."""
    model = CQL(n_epochs=1, mdp_dataset_builder=MdpDatasetBuilder(top_k=1))
    model.fit(log)
    recs = model.predict(log, k=1).toPandas()

    # for asserted users we know which items are not in the training set,
    # so check that one of them is recommended
    assert recs.loc[recs["user_idx"] == 0, "item_idx"].iloc[0] == 3
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] in [1, 2]
    assert recs.loc[recs["user_idx"] == 2, "item_idx"].iloc[0] == 3


def test_recommend_correct_number_of_items(log: DataFrame):
    """Test that fit/predict_pairs works, and that the model outputs correct number of items."""
    top_k = 3
    model = CQL(n_epochs=1, mdp_dataset_builder=MdpDatasetBuilder(top_k=top_k))
    model.fit(log)

    train_user_item_pairs = log.select("user_idx", "item_idx")
    recs = model.predict_pairs(train_user_item_pairs, log, k=top_k).toPandas()

    # we know how many items are in the training set for each user, just check this number
    assert np.count_nonzero(recs["user_idx"] == 0) == 3
    assert np.count_nonzero(recs["user_idx"] == 1) == 2
    assert np.count_nonzero(recs["user_idx"] == 2) == 3


def test_serialize_deserialize_policy(log: DataFrame):
    """Test that serializing and deserializing the policy does not change relevance predictions."""
    model = CQL(n_epochs=1, mdp_dataset_builder=MdpDatasetBuilder(top_k=1))
    model.fit(log)

    # arbitrary batch of user-item pairs as we test exact relevance for each one
    user_item_batch = np.array([
        [0, 0], [0, 1], [0, 2], [0, 3],
        [1, 0], [1, 1], [1, 2], [1, 3],
    ])

    # predict user-item relevance with the original model
    relevance = model.model.predict(user_item_batch)

    # predict user-item relevance with the serialized-then-restored policy
    restored_policy = model._deserialize_policy(model._serialize_policy())
    restored_relevance = model._predict_relevance_with_policy(restored_policy, user_item_batch)

    assert restored_relevance == approx(relevance)


def test_mdp_dataset_builder(log: DataFrame):
    """Test MDP dataset preparation is correct."""
    mdp_dataset = MdpDatasetBuilder(top_k=1, action_randomization_scale=1e-9).build(log)

    # we test only users {0, 1} as for the rest log has non-deterministic order (see log dates)
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

    # as we do not guarantee and require that MDP preparation should keep ints as ints
    # and keeps floats exactly the same, we use approx for all equality checks
    assert mdp_dataset.observations[:n] == approx(gt_observations)
    # larger approx to take into account action randomization noise added to the MDP actions
    assert mdp_dataset.actions[:n].flatten() == approx(gt_actions, abs=1e-2)
    assert mdp_dataset.rewards[:n] == approx(gt_rewards)
    assert mdp_dataset.terminals[:n] == approx(gt_terminals)
