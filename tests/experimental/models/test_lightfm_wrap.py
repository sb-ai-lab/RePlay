# pylint: disable-all
from datetime import datetime

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

from pyspark.sql import functions as sf

from replay.data import get_schema
from replay.experimental.models import LightFMWrap
from replay.experimental.models.base_rec import HybridRecommender, UserRecommender
from replay.experimental.scenarios.two_stages.two_stages_scenario import get_first_level_model_features
from replay.experimental.utils.model_handler import save
from replay.utils.model_handler import load
from tests.utils import log, log_to_pred, long_log_with_features, spark, sparkDataFrameEqual, user_features

SEED = 123


def fit_predict_selected(model, train_log, inf_log, user_features, users):
    kwargs = {}
    if isinstance(model, (HybridRecommender, UserRecommender)):
        kwargs = {"user_features": user_features}
    model.fit(train_log, **kwargs)
    return model.predict(log=inf_log, users=users, k=1, **kwargs)


@pytest.fixture
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, 0, date, 1.0],
            [1, 0, date, 1.0],
            [2, 1, date, 2.0],
            [2, 1, date, 2.0],
            [1, 1, date, 2.0],
            [2, 2, date, 2.0],
            [0, 2, date, 2.0],
        ],
        schema=get_schema("user_idx", "item_idx", "timestamp", "relevance"),
    )


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame(
        [(0, 2.0, 5.0), (1, 0.0, -5.0), (4, 4.0, 3.0)]
    ).toDF("user_idx", "user_feature_1", "user_feature_2")


@pytest.fixture
def item_features(spark):
    return spark.createDataFrame([(0, 4.0, 5.0), (1, 5.0, 4.0)]).toDF(
        "item_idx", "item_feature_1", "item_feature_2"
    )


@pytest.fixture
def model():
    model = LightFMWrap(no_components=1, random_state=42, loss="bpr")
    model.num_threads = 1
    return model


@pytest.mark.experimental
def test_equal_preds(long_log_with_features, tmp_path):
    path = (tmp_path / "test").resolve()
    model = LightFMWrap()
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    loaded_model = load(path, LightFMWrap)
    new_pred = loaded_model.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.experimental
def test_predict(log, user_features, item_features, model):
    model.fit(log, user_features, item_features)
    pred = model.predict(
        log=log,
        k=1,
        user_features=user_features,
        item_features=item_features,
        filter_seen_items=True,
    )
    assert list(pred.toPandas().sort_values("user_idx")["item_idx"]) == [
        1,
        2,
        0,
    ]


@pytest.mark.experimental
def test_predict_no_user_features(log, item_features, model):
    model.fit(log, None, item_features)
    assert model.can_predict_cold_items
    assert not model.can_predict_cold_users
    pred = model.predict(
        log=log,
        k=1,
        user_features=None,
        item_features=item_features,
        filter_seen_items=True,
    )
    assert list(pred.toPandas().sort_values("user_idx")["item_idx"]) == [
        1,
        2,
        0,
    ]


@pytest.mark.experimental
def test_predict_pairs(log, user_features, item_features, model):
    try:
        model.fit(
            log.filter(sf.col("user_idx") != 0),
            user_features.filter(sf.col("user_idx") != 0),
            item_features,
        )
        pred = model.predict_pairs(
            log.filter(sf.col("user_idx") == 0).select("user_idx", "item_idx"),
            user_features=user_features,
            item_features=item_features,
        )
        assert pred.count() == 2
        assert pred.select("user_idx").distinct().collect()[0][0] == 0
        pred = model.predict_pairs(
            log.filter(sf.col("user_idx") == 1).select("user_idx", "item_idx"),
            user_features=user_features,
            item_features=item_features,
        )
        assert pred.count() == 2
        assert pred.select("user_idx").distinct().collect()[0][0] == 1
    except:  # noqa
        pytest.fail()


@pytest.mark.experimental
def test_raises_fit(log, user_features, item_features, model):
    with pytest.raises(ValueError, match="features for .*"):
        model.fit(
            log.filter(sf.col("user_idx") != 0),
            user_features.filter(sf.col("user_idx") != 1),
            item_features,
        )


@pytest.mark.experimental
def test_raises_predict(log, item_features, model):
    with pytest.raises(
        ValueError, match="Item features are missing for predict"
    ):
        model.fit(log, None, item_features)
        _ = model.predict_pairs(
            log.select("user_idx", "item_idx"),
            user_features=None,
            item_features=None,
        )


def _fit_predict_compare_features(
    model, log, user_features, user_features_filtered, item_features, test_ids
):
    model.fit(
        log, user_features=user_features_filtered, item_features=item_features
    )

    pred_for_test = (
        model.predict_pairs(
            test_ids.select("user_idx", "item_idx"),
            log,
            user_features=user_features,
            item_features=item_features,
        )
        .select("relevance")
        .collect()[0][0]
    )
    row_dict = (
        get_first_level_model_features(
            model,
            test_ids,
            user_features=user_features,
            item_features=item_features,
        )
        .collect()[0]
        .asDict()
    )
    assert np.isclose(
        row_dict["_if_0"] * row_dict["_uf_0"]
        + row_dict["_user_bias"]
        + row_dict["_item_bias"],
        pred_for_test,
    )


@pytest.mark.experimental
def test_enrich_with_features(log, user_features, item_features, model):
    test_pair = log.filter(
        (sf.col("item_idx") == 1) & (sf.col("user_idx") == 1)
    )

    for user_f, item_f in [[None, None], [user_features, item_features]]:
        _fit_predict_compare_features(
            model, log, user_f, user_f, item_f, test_pair
        )
        if item_f is not None:
            _fit_predict_compare_features(
                model,
                log.filter(sf.col("user_idx") != 1),
                user_f,
                user_f.filter(sf.col("user_idx") != 1),
                item_f,
                test_pair,
            )


@pytest.mark.experimental
def test_predict_pairs_warm_items_only(log, log_to_pred):
    model = LightFMWrap(random_state=SEED)
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
    model = LightFMWrap(random_state=SEED)
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
    model = LightFMWrap(random_state=SEED)
    model.fit(log)
    model.predict(log.limit(0), 1)


@pytest.mark.experimental
def test_predict_new_users(long_log_with_features, user_features):
    model = LightFMWrap(random_state=SEED, no_components=4)
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
def test_predict_cold_users(long_log_with_features, user_features):
    model = LightFMWrap(random_state=SEED, no_components=4)
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        user_features=user_features.drop("gender"),
        users=[0],
    )
    assert pred.count() == 1
    assert pred.collect()[0][0] == 0


@pytest.mark.experimental
def test_predict_cold_and_new_filter_out(long_log_with_features):
    model = LightFMWrap()
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features,
        user_features=None,
        users=[0, 3],
    )

    if isinstance(model, LightFMWrap) or not model.can_predict_cold_users:
        assert pred.count() == 0
    else:
        assert 1 <= pred.count() <= 2
