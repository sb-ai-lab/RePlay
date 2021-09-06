# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
from datetime import datetime

import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.constants import LOG_SCHEMA
from replay.models import (
    ALSWrap,
    ADMMSLIM,
    KNN,
    LightFMWrap,
    NeuroMF,
    PopRec,
    SLIM,
    MultVAE,
    Word2VecRec,
)

from tests.utils import spark, log, sparkDataFrameEqual

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
        Word2VecRec(seed=SEED),
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
    [ADMMSLIM(seed=SEED), KNN(), SLIM(seed=SEED), Word2VecRec(seed=SEED)],
    ids=["admm_slim", "knn", "slim", "word2vec",],
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


# for Neighbour recommenders and ALS
@pytest.mark.parametrize(
    "model",
    [ALSWrap(seed=SEED), ADMMSLIM(seed=SEED), KNN(), SLIM(seed=SEED)],
    ids=["als", "admm_slim", "knn", "slim"],
)
def test_get_nearest_items(log, model):
    model.fit(log.filter(sf.col("item_id") != "item4"))
    # cosine
    res = model.get_nearest_items(
        items=["item1", "item2"], k=2, metric="cosine_similarity"
    )

    assert res.count() == 4
    assert set(res.toPandas().to_dict()["item_id"].values()) == {
        "item1",
        "item2",
    }

    # squared
    res = model.get_nearest_items(
        items=["item1", "item2"], k=1, metric="squared_distance"
    )
    assert res.count() == 2

    # filter neighbours
    res = model.get_nearest_items(
        items=["item1", "item2"],
        k=4,
        metric="squared_distance",
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
            items=["item1", "item2"], k=2, metric="squared_distance"
        )
