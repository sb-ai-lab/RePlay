# pylint: disable-all
import numpy as np
import pytest
from _pytest.python_api import approx
from pytest import approx

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

from pyspark.sql import functions as sf

from replay.experimental.models import CQL
from replay.experimental.models.base_rec import HybridRecommender, UserRecommender
from replay.experimental.models.cql import MdpDatasetBuilder
from replay.utils import SparkDataFrame
from tests.utils import log, log_to_pred, long_log_with_features, spark, sparkDataFrameEqual, user_features


def fit_predict_selected(model, train_log, inf_log, user_features, users):
    kwargs = {}
    if isinstance(model, (HybridRecommender, UserRecommender)):
        kwargs = {"user_features": user_features}
    model.fit(train_log, **kwargs)
    return model.predict(log=inf_log, users=users, k=1, **kwargs)


@pytest.mark.experimental
def test_predict_filters_out_seen_items(log: SparkDataFrame):
    """Test that fit/predict works, and that the model correctly filters out seen items."""
    model = CQL(n_epochs=1, mdp_dataset_builder=MdpDatasetBuilder(top_k=1))
    model.fit(log)
    recs = model.predict(log, k=1).toPandas()

    # for asserted users we know which items are not in the training set,
    # so check that one of them is recommended
    assert recs.loc[recs["user_idx"] == 0, "item_idx"].iloc[0] == 3
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] in [1, 2]
    assert recs.loc[recs["user_idx"] == 2, "item_idx"].iloc[0] == 3


@pytest.mark.experimental
def test_recommend_correct_number_of_items(log: SparkDataFrame):
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


@pytest.mark.experimental
def test_serialize_deserialize_policy(log: SparkDataFrame):
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


@pytest.mark.experimental
def test_mdp_dataset_builder(log: SparkDataFrame):
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


@pytest.mark.experimental
def test_predict_pairs_warm_items_only(log, log_to_pred):
    model = CQL(n_epochs=1, mdp_dataset_builder=MdpDatasetBuilder(top_k=3), batch_size=512)
    model.fit(log)
    recs = model.predict(
        log.unionByName(log_to_pred),
        k=3,
        users=log_to_pred.select("user_idx").distinct(),
        items=log_to_pred.select("item_idx").distinct(),
        filter_seen_items=False,
    )

    pairs_pred = model.predict_pairs(
        pairs=log_to_pred.select("user_idx", "item_idx"),
        log=log.unionByName(log_to_pred),
    )

    condition = ~sf.col("item_idx").isin([4, 5])
    if not model.can_predict_cold_users:
        condition = condition & (sf.col("user_idx") != 4)

    sparkDataFrameEqual(
        pairs_pred.select("user_idx", "item_idx"),
        log_to_pred.filter(condition).select("user_idx", "item_idx"),
    )

    recs_joined = (
        pairs_pred.withColumnRenamed("relevance", "pairs_relevance")
        .join(recs, on=["user_idx", "item_idx"], how="left")
        .sort("user_idx", "item_idx")
    )

    assert np.allclose(
        recs_joined.select("relevance").toPandas().to_numpy(),
        recs_joined.select("pairs_relevance").toPandas().to_numpy(),
    )


@pytest.mark.experimental
def test_predict_new_users(long_log_with_features, user_features):
    model = CQL(n_epochs=1, mdp_dataset_builder=MdpDatasetBuilder(top_k=1), batch_size=512)
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features,
        user_features=user_features.drop("gender"),
        users=[0],
    )
    assert pred.count() == 1
    assert pred.collect()[0][0] == 0


@pytest.mark.experimental
def test_predict_cold_and_new_filter_out(long_log_with_features):
    model = CQL(n_epochs=1, mdp_dataset_builder=MdpDatasetBuilder(top_k=3), batch_size=512)
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features,
        user_features=None,
        users=[0, 3],
    )
    # assert new/cold users are filtered out in `predict`
    if not model.can_predict_cold_users:
        assert pred.count() == 0
    else:
        assert 1 <= pred.count() <= 2
