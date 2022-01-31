# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
from datetime import datetime

import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.constants import LOG_SCHEMA
from replay.models import (
    ALSWrap,
    ADMMSLIM,
    ClusterRec,
    KNN,
    LightFMWrap,
    NeuroMF,
    PopRec,
    RandomRec,
    SLIM,
    MultVAE,
    Word2VecRec,
)
from replay.models.base_rec import HybridRecommender, UserRecommender

from tests.utils import (
    spark,
    log,
    long_log_with_features,
    user_features,
    sparkDataFrameEqual,
)

SEED = 123


@pytest.fixture
def log_to_pred(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item4", datetime(2019, 9, 12), 3.0],
            ["user1", "item5", datetime(2019, 9, 13), 2.0],
            ["user2", "item2", datetime(2019, 9, 17), 1.0],
            ["user2", "item6", datetime(2019, 9, 14), 4.0],
            ["user3", "item4", datetime(2019, 9, 15), 3.0],
            ["user5", "item2", datetime(2019, 9, 15), 3.0],
            ["user5", "item3", datetime(2019, 9, 15), 3.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.mark.parametrize(
    "model",
    [
        ALSWrap(seed=SEED),
        ADMMSLIM(seed=SEED),
        KNN(),
        LightFMWrap(random_state=SEED),
        MultVAE(),
        NeuroMF(),
        SLIM(seed=SEED),
        Word2VecRec(seed=SEED, min_count=0),
        PopRec(),
    ],
    ids=[
        "als",
        "admm_slim",
        "knn",
        "lightfm",
        "multvae",
        "neuromf",
        "slim",
        "word2vec",
        "poprec",
    ],
)
def test_predict_pairs_warm_only(log, log_to_pred, model):
    model.fit(log)
    recs = model.predict(
        log.unionByName(log_to_pred),
        k=3,
        users=log_to_pred.select("user_id").distinct(),
        items=log_to_pred.select("item_id").distinct(),
        filter_seen_items=False,
    )

    pairs_pred = model.predict_pairs(
        pairs=log_to_pred.select("user_id", "item_id"),
        log=log.unionByName(log_to_pred),
    )

    condition = ~sf.col("item_id").isin(["item5", "item6"])
    if not model.can_predict_cold_users:
        condition = condition & (sf.col("user_id") != "user5")

    sparkDataFrameEqual(
        pairs_pred.select("user_id", "item_id"),
        log_to_pred.filter(condition).select("user_id", "item_id"),
    )

    recs_joined = (
        pairs_pred.withColumnRenamed("relevance", "pairs_relevance")
        .join(recs, on=["user_id", "item_id"], how="left")
        .sort("user_id", "item_id")
    )

    assert np.allclose(
        recs_joined.select("relevance").toPandas().to_numpy(),
        recs_joined.select("pairs_relevance").toPandas().to_numpy(),
    )


@pytest.mark.parametrize(
    "model",
    [
        ADMMSLIM(seed=SEED),
        KNN(),
        SLIM(seed=SEED),
        Word2VecRec(seed=SEED, min_count=0),
    ],
    ids=[
        "admm_slim",
        "knn",
        "slim",
        "word2vec",
    ],
)
def test_predict_pairs_raises(log, model):
    with pytest.raises(ValueError, match="log is not provided,.*"):
        model.fit(log)
        model.predict_pairs(log.select("user_id", "item_id"))


def test_predict_pairs_raises_pairs_format(log):
    model = ALSWrap(seed=SEED)
    with pytest.raises(ValueError, match="pairs must be a dataframe with .*"):
        model.fit(log)
        model.predict_pairs(log, log)


# for NeighbourRec and ItemVectorModel
@pytest.mark.parametrize(
    "model, metric",
    [
        (ALSWrap(seed=SEED), "euclidean_distance_sim"),
        (ALSWrap(seed=SEED), "dot_product"),
        (ALSWrap(seed=SEED), "cosine_similarity"),
        (Word2VecRec(seed=SEED, min_count=0), "cosine_similarity"),
        (ADMMSLIM(seed=SEED), None),
        (KNN(), None),
        (SLIM(seed=SEED), None),
    ],
    ids=[
        "als_euclidean",
        "als_dot",
        "als_cosine",
        "w2v_cosine",
        "admm_slim",
        "knn",
        "slim",
    ],
)
def test_get_nearest_items(log, model, metric):
    model.fit(log.filter(sf.col("item_id") != "item4"))
    res = model.get_nearest_items(items=["item1", "item2"], k=2, metric=metric)

    assert res.count() == 4
    assert set(res.toPandas().to_dict()["item_id"].values()) == {
        "item1",
        "item2",
    }

    res = model.get_nearest_items(items=["item1", "item2"], k=1, metric=metric)
    assert res.count() == 2

    # filter neighbours
    res = model.get_nearest_items(
        items=["item1", "item2"],
        k=4,
        metric=metric,
        candidates=["item1", "item4"],
    )
    assert res.count() == 1
    assert (
        len(
            set(res.toPandas().to_dict()["item_id"].values()).difference(
                {"item1", "item2"}
            )
        )
        == 0
    )


def test_nearest_items_raises(log):
    model = PopRec()
    model.fit(log.filter(sf.col("item_id") != "item4"))
    with pytest.raises(
        ValueError, match=r"Distance metric is required to get nearest items.*"
    ):
        model.get_nearest_items(items=["item1", "item2"], k=2, metric=None)

    with pytest.raises(
        ValueError,
        match=r"Use models with attribute 'can_predict_item_to_item' set to True.*",
    ):
        model.get_nearest_items(
            items=["item1", "item2"], k=2, metric="cosine_similarity"
        )

        with pytest.raises(
            ValueError,
            match=r"Use models with attribute 'can_predict_item_to_item' set to True.*",
        ):
            model.get_nearest_items(
                items=["item1", "item2"], k=2, metric="cosine_similarity"
            )


def test_filter_seen(log):
    model = PopRec()
    # filter seen works with empty log to filter (cold_user)
    model.fit(log.filter(sf.col("user_id") != "user1"))
    pred = model.predict(log=log, users=["user5"], k=5)
    assert pred.count() == 4

    # filter seen works with log not presented during training (for user1)
    pred = model.predict(log=log, users=["user1"], k=5)
    assert pred.count() == 1

    # filter seen turns off
    pred = model.predict(
        log=log, users=["user1"], k=5, filter_seen_items=False
    )
    assert pred.count() == 4


def fit_predict_selected(model, train_log, inf_log, user_features, users):
    kwargs = {}
    if isinstance(model, (HybridRecommender, UserRecommender)):
        kwargs = {"user_features": user_features}
    model.fit(train_log, **kwargs)
    return model.predict(log=inf_log, users=users, k=1, **kwargs)


@pytest.mark.parametrize(
    "model",
    [
        ADMMSLIM(seed=SEED),
        ClusterRec(num_clusters=2),
        KNN(),
        LightFMWrap(random_state=SEED, no_components=4),
        MultVAE(),
        SLIM(seed=SEED),
        PopRec(),
        RandomRec(seed=SEED),
        Word2VecRec(seed=SEED, min_count=0),
    ],
    ids=[
        "admm_slim",
        "cluster",
        "knn",
        "lightfm",
        "multvae",
        "slim",
        "pop_rec",
        "random_rec",
        "word2vec",
    ],
)
def test_predict_new_users(model, long_log_with_features, user_features):
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_id") != "u1"),
        inf_log=long_log_with_features,
        user_features=user_features.drop("gender"),
        users=["u1"],
    )
    assert pred.count() == 1
    assert pred.collect()[0][0] == "u1"


@pytest.mark.parametrize(
    "model",
    [
        ClusterRec(num_clusters=2),
        LightFMWrap(random_state=SEED, no_components=4),
        PopRec(),
        RandomRec(seed=SEED),
    ],
    ids=[
        "cluster",
        "lightfm",
        "pop_rec",
        "random_rec",
    ],
)
def test_predict_cold_users(model, long_log_with_features, user_features):
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_id") != "u1"),
        inf_log=long_log_with_features.filter(sf.col("user_id") != "u1"),
        user_features=user_features.drop("gender"),
        users=["u1"],
    )
    assert pred.count() == 1
    assert pred.collect()[0][0] == "u1"


@pytest.mark.parametrize(
    "model",
    [
        ALSWrap(rank=2, seed=SEED),
        KNN(),
        LightFMWrap(),
        MultVAE(),
        NeuroMF(),
        SLIM(seed=SEED),
        Word2VecRec(seed=SEED, min_count=0),
    ],
    ids=[
        "als",
        "knn",
        "lightfm_no_feat",
        "multvae",
        "neuromf",
        "slim",
        "word2vec",
    ],
)
def test_predict_cold_and_new_filter_out(model, long_log_with_features):
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_id") != "u1"),
        inf_log=long_log_with_features,
        user_features=None,
        users=["u1", "cold_user"],
    )
    # assert new/cold users are filtered out in `predict`
    if isinstance(model, LightFMWrap) or not model.can_predict_cold_users:
        assert pred.count() == 0
    else:
        assert 1 <= pred.count() <= 2
