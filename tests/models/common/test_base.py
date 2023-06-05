# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest
import implicit
import numpy as np

from pyspark.sql import functions as sf

from replay.constants import LOG_SCHEMA
from replay.models import (
    ADMMSLIM,
    ALSWrap,
    AssociationRulesItemRec,
    ImplicitWrap,
    ItemKNN,
    LightFMWrap,
    PopRec,
    RandomRec,
    SLIM,
    ThompsonSampling,
    UCB,
    UserPopRec,
    Wilson,
    Word2VecRec,
)
from replay.model_handler import save, load
from tests.utils import log, pos_neg_log, SEED, spark, sparkDataFrameEqual


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


@pytest.mark.parametrize(
    "model",
    [
        ADMMSLIM(),
        ALSWrap(),
        AssociationRulesItemRec(min_item_count=1, min_pair_count=0),
        ImplicitWrap(implicit.als.AlternatingLeastSquares()),
        ItemKNN(),
        LightFMWrap(),
        PopRec(),
        RandomRec(seed=1),
        SLIM(),
        UserPopRec(),
        Word2VecRec(),
    ],
)
def test_save_load(log, model, tmp_path):
    path = (tmp_path / "test").resolve()
    model.fit(log)
    base_pred = model.predict(log, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(log, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.parametrize(
    "model",
    [ThompsonSampling(), UCB(), Wilson()],
    ids=["thompson", "ucb", "wilson"]
)
def test_save_load_pos_neg_log(model, pos_neg_log, tmp_path):
    path = (tmp_path / "model").resolve()
    model.fit(pos_neg_log)
    base_pred = model.predict(pos_neg_log, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(pos_neg_log, 5)
    sparkDataFrameEqual(base_pred, new_pred)


# for ItemVectorModel and NeighbourRec
@pytest.mark.parametrize(
    "model, metric",
    [
        (ALSWrap(seed=SEED), "euclidean_distance_sim"),
        (ALSWrap(seed=SEED), "dot_product"),
        (ALSWrap(seed=SEED), "cosine_similarity"),
        (Word2VecRec(seed=SEED, min_count=0), "cosine_similarity"),
        (ADMMSLIM(seed=SEED), None),
        (AssociationRulesItemRec(min_item_count=1, min_pair_count=0), "lift"),
        (
            AssociationRulesItemRec(min_item_count=1, min_pair_count=0),
            "confidence",
        ),
        (
            AssociationRulesItemRec(min_item_count=1, min_pair_count=0),
            "confidence_gain",
        ),
        (ItemKNN(), None),
        (SLIM(seed=SEED), None),
    ],
    ids=[
        "als_euclidean",
        "als_dot",
        "als_cosine",
        "w2v_cosine",
        "admm_slim",
        "association_rules_lift",
        "association_rules_confidence",
        "association_rules_confidence_gain",
        "knn",
        "slim",
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
def test_get_nearest_items_raises(log, metric):
    model = ALSWrap()
    model.fit(log)
    with pytest.raises(
        ValueError, match=r"Select one of the valid distance metrics.*"
    ):
        model.get_nearest_items(items=[0, 1], k=2, metric=metric)

    model = Word2VecRec()
    model.fit(log)
    with pytest.raises(
        ValueError, match=r"Select one of the valid distance metrics.*"
    ):
        model.get_nearest_items(items=[0, 1], k=2, metric=metric)

    # AssociationRulesItemRec has overwritten get_nearest_items
    model = AssociationRulesItemRec(min_item_count=1, min_pair_count=0)
    model.fit(log.filter(sf.col("item_idx") != 3))
    with pytest.raises(
        ValueError, match=r"Select one of the valid distance metrics.*"
    ):
        model.get_nearest_items(items=[0, 1], k=2, metric=metric)


@pytest.mark.parametrize(
    "new_metric",
    ["lift", "absent"],
)
def test_similarity_metric_change(log, new_metric):
    model = AssociationRulesItemRec(
        min_item_count=1,
        min_pair_count=0,
        similarity_metric="confidence"
    )
    if new_metric in model.item_to_item_metrics:
        model.fit(log)
        model.similarity_metric = new_metric
    else:
        with pytest.raises(
            ValueError, match=r"Select one of the valid metrics.*"
        ):
            model.fit(log)
            model.similarity_metric = new_metric


@pytest.mark.parametrize(
    "model",
    [
        ADMMSLIM(),
        ItemKNN(),
        SLIM(),
    ],
    ids=[
        "admm_slim",
        "knn",
        "slim",
    ],
)
def test_similarity_metric_change_raises(log, model):
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
