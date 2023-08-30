# pylint: disable-all
from datetime import datetime

import numpy as np
import pytest
from pyspark.sql import functions as sf

from replay.data import LOG_SCHEMA
from replay.models import LightFMWrap
from replay.scenarios.two_stages.two_stages_scenario import (
    get_first_level_model_features,
)
from tests.utils import spark


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
        schema=LOG_SCHEMA,
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


def test_predict_no_user_features(log, user_features, item_features, model):
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


def test_raises_fit(log, user_features, item_features, model):
    with pytest.raises(ValueError, match="features for .*"):
        model.fit(
            log.filter(sf.col("user_idx") != 0),
            user_features.filter(sf.col("user_idx") != 1),
            item_features,
        )


def test_raises_predict(log, user_features, item_features, model):
    with pytest.raises(
        ValueError, match="Item features are missing for predict"
    ):
        model.fit(log, None, item_features)
        pred = model.predict_pairs(
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
