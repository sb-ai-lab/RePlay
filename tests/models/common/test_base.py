# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.constants import LOG_SCHEMA
from replay.models import (
    ADMMSLIM,
    ALSWrap,
    AssociationRulesItemRec,
    ItemKNN,
    PopRec,
    SLIM,
    ThompsonSampling,
    UCB,
    Wilson,
    Word2VecRec,
)
from tests.utils import log, pos_neg_log, SEED, spark


def test_filter_seen(log):
    model = PopRec()
    # filter seen works with empty log to filter (cold_user)
    model.fit(log.filter(sf.col("user_idx") != 0))
    pred = model.predict(log=log, users=[3], k=5)
    assert pred.count() == 2

    # filter seen works with log not presented during training (for user1)
    pred = model.predict(log=log, users=[0], k=5)
    assert pred.count() == 1

    # filter seen turns off
    pred = model.predict(log=log, users=[0], k=5, filter_seen_items=False)
    assert pred.count() == 4


# for NeighbourRec and ItemVectorModel
@pytest.mark.parametrize(
    "model, metric",
    [
        (ALSWrap(seed=SEED), "euclidean_distance_sim"),
        (ALSWrap(seed=SEED), "dot_product"),
        (ALSWrap(seed=SEED), "cosine_similarity"),
        (Word2VecRec(seed=SEED, min_count=0), "cosine_similarity"),
        (ADMMSLIM(seed=SEED), None),
        (ItemKNN(), None),
        (SLIM(seed=SEED), None),
        (AssociationRulesItemRec(min_item_count=1, min_pair_count=0), "lift"),
        (
            AssociationRulesItemRec(min_item_count=1, min_pair_count=0),
            "confidence",
        ),
        (
            AssociationRulesItemRec(min_item_count=1, min_pair_count=0),
            "confidence_gain",
        ),
    ],
    ids=[
        "als_euclidean",
        "als_dot",
        "als_cosine",
        "w2v_cosine",
        "admm_slim",
        "knn",
        "slim",
        "association_rules_lift",
        "association_rules_confidence",
        "association_rules_confidence_gain",
    ],
)
def test_get_nearest_items(log, model, metric):
    model.fit(log.filter(sf.col("item_idx") != 3))
    res = model.get_nearest_items(items=[0, 1], k=2, metric=metric)

    assert res.count() == 4
    assert set(res.toPandas().to_dict()["item_idx"].values()) == {
        0,
        1,
    }

    res = model.get_nearest_items(items=[0, 1], k=1, metric=metric)
    assert res.count() == 2

    # filter neighbours
    res = model.get_nearest_items(
        items=[0, 1],
        k=4,
        metric=metric,
        candidates=[0, 3],
    )
    assert res.count() == 1
    assert (
        len(
            set(res.toPandas().to_dict()["item_idx"].values()).difference(
                {0, 1}
            )
        )
        == 0
    )


@pytest.mark.parametrize("metric", ["absent", None])
def test_nearest_items_raises(log, metric):
    model = AssociationRulesItemRec()
    model.fit(log.filter(sf.col("item_idx") != 3))
    with pytest.raises(
        ValueError, match=r"Select one of the valid distance metrics.*"
    ):
        model.get_nearest_items(items=[0, 1], k=2, metric=metric)
    model = ALSWrap()
    model.fit(log)
    with pytest.raises(
        ValueError, match=r"Select one of the valid distance metrics.*"
    ):
        model.get_nearest_items(items=[0, 1], k=2, metric=metric)


@pytest.mark.parametrize(
    "model",
    [
        ItemKNN(),
        SLIM(seed=SEED),
    ],
    ids=[
        "knn",
        "slim",
    ],
)
def test_similarity_metric_raises(log, model):
    with pytest.raises(
        ValueError,
        match="This class does not support changing similarity metrics",
    ):
        model.fit(log)
        model.similarity_metric = "some"


@pytest.mark.parametrize(
    "model",
    [
        PopRec(),
        ThompsonSampling(),
        UCB(),
        Wilson(),
    ],
    ids=[
        "pop_rec",
        "thompson",
        "UCB",
        "wilson",
    ],
)
def test_item_popularity(model, pos_neg_log):
    model.fit(pos_neg_log)
    assert (
        model.item_popularity.count()
        == pos_neg_log.select("item_idx").distinct().count()
    )
    model.item_popularity.count()
