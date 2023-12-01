# pylint: disable-all
from datetime import datetime

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

from pyspark.sql import functions as sf

from replay.data import get_schema
from replay.experimental.models import ADMMSLIM
from replay.experimental.models.base_rec import HybridRecommender, UserRecommender
from replay.experimental.utils.model_handler import save
from replay.utils.model_handler import load
from tests.utils import log, log_to_pred, long_log_with_features, spark, sparkDataFrameEqual, user_features

SEED = 123
INTERACTIONS_SCHEMA = get_schema("user_idx", "item_idx", "timestamp", "relevance")


@pytest.mark.experimental
def test_equal_preds(long_log_with_features, tmp_path):
    path = (tmp_path / "test").resolve()
    model = ADMMSLIM()
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    loaded_model = load(path, ADMMSLIM)
    new_pred = loaded_model.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def fit_predict_selected(model, train_log, inf_log, user_features, users):
    kwargs = {}
    if isinstance(model, (HybridRecommender, UserRecommender)):
        kwargs = {"user_features": user_features}
    model.fit(train_log, **kwargs)
    return model.predict(log=inf_log, users=users, k=1, **kwargs)


@pytest.fixture
def simple_log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, 0, date, 1.0],
            [1, 0, date, 1.0],
            [2, 1, date, 2.0],
            [1, 1, date, 2.0],
            [2, 2, date, 2.0],
            [0, 2, date, 2.0],
            [3, 0, date, 2.0],
        ],
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture
def model():
    return ADMMSLIM(1, 10, 42)


@pytest.mark.experimental
def test_fit(simple_log, model):
    model.fit(simple_log)
    assert np.allclose(
        model.similarity.toPandas().to_numpy(),
        [
            (0, 1, 0.03095617860316846),
            (0, 2, 0.030967752554031502),
            (1, 0, 0.031891083964224354),
            (1, 2, 0.1073860741574666),
            (2, 0, 0.031883667509449376),
            (2, 1, 0.10739028463512135),
        ],
    )


@pytest.mark.experimental
def test_predict(simple_log, model):
    model.fit(simple_log)
    recs = model.predict(simple_log, k=1)
    assert recs.count() == 4


@pytest.mark.experimental
@pytest.mark.parametrize(
    "lambda_1,lambda_2", [(0.0, 0.0), (-0.1, 0.1), (0.1, -0.1)]
)
def test_exceptions(lambda_1, lambda_2):
    with pytest.raises(ValueError):
        ADMMSLIM(lambda_1, lambda_2)


@pytest.mark.experimental
def test_predict_pairs_warm_items_only(log, log_to_pred):
    model = ADMMSLIM(seed=SEED)
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
def test_predict_pairs_k(log):
    model = ADMMSLIM(seed=SEED)
    model.fit(log)

    pairs_pred_k = model.predict_pairs(
        pairs=log.select("user_idx", "item_idx"),
        log=log,
        k=1,
    )

    pairs_pred = model.predict_pairs(
        pairs=log.select("user_idx", "item_idx"),
        log=log,
        k=None,
    )

    assert (
        pairs_pred_k.groupBy("user_idx")
        .count()
        .filter(sf.col("count") > 1)
        .count()
        == 0
    )

    assert (
        pairs_pred.groupBy("user_idx")
        .count()
        .filter(sf.col("count") > 1)
        .count()
        > 0
    )


@pytest.mark.experimental
def test_predict_empty_log(log):
    model = ADMMSLIM(seed=SEED)
    model.fit(log)
    model.predict(log.limit(0), 1)


@pytest.mark.experimental
def test_predict_pairs_raises(log):
    model = ADMMSLIM(seed=SEED)
    with pytest.raises(ValueError, match="log is not provided,.*"):
        model.fit(log)
        model.predict_pairs(log.select("user_idx", "item_idx"))


@pytest.mark.experimental
def test_get_nearest_items(log):
    model = ADMMSLIM(seed=SEED)
    model.fit(log.filter(sf.col("item_idx") != 3))
    res = model.get_nearest_items(items=[0, 1], k=2, metric=None)

    assert res.count() == 4
    assert set(res.toPandas().to_dict()["item_idx"].values()) == {
        0,
        1,
    }

    res = model.get_nearest_items(items=[0, 1], k=1, metric=None)
    assert res.count() == 2

    # filter neighbours
    res = model.get_nearest_items(
        items=[0, 1],
        k=4,
        metric=None,
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


@pytest.mark.experimental
def test_predict_new_users(long_log_with_features, user_features):
    model = ADMMSLIM(seed=SEED)
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features,
        user_features=user_features.drop("gender"),
        users=[0],
    )
    assert pred.count() == 1
    assert pred.collect()[0][0] == 0
