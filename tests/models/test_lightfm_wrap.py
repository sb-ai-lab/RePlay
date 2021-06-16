# pylint: disable-all
from datetime import datetime

import pytest
from pyspark.sql import functions as sf

from replay.constants import LOG_SCHEMA
from replay.models import LightFMWrap
from tests.utils import spark


@pytest.fixture
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            ["u1", "i1", date, 1.0],
            ["u2", "i1", date, 1.0],
            ["u3", "i3", date, 2.0],
            ["u3", "i3", date, 2.0],
            ["u2", "i3", date, 2.0],
            ["u3", "i4", date, 2.0],
            ["u1", "i4", date, 2.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame(
        [("u1", 2.0, 5.0), ("u2", 0.0, -5.0), ("u5", 4.0, 3.0)]
    ).toDF("user_id", "user_feature_1", "user_feature_2")


@pytest.fixture
def item_features(spark):
    return spark.createDataFrame([("i1", 4.0, 5.0), ("i3", 5.0, 4.0)]).toDF(
        "item_id", "item_feature_1", "item_feature_2"
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
    assert list(pred.toPandas().sort_values("user_id")["item_id"]) == [
        "i3",
        "i4",
        "i1",
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
    assert list(pred.toPandas().sort_values("user_id")["item_id"]) == [
        "i3",
        "i4",
        "i1",
    ]


# predict для пар с фичами
def test_predict_pairs(log, user_features, item_features, model):
    try:
        model.fit(
            log.filter(sf.col("user_id") != "u1"),
            user_features.filter(sf.col("user_id") != "u1"),
            item_features,
        )
        # предсказываем для холодного пользователя
        pred = model.predict_pairs(
            log.filter(sf.col("user_id") == "u1").select("user_id", "item_id"),
            user_features=user_features,
            item_features=item_features,
        )
        assert pred.count() == 2
        assert pred.select("user_id").distinct().collect()[0][0] == "u1"
        # предсказываем для теплого пользователя
        pred = model.predict_pairs(
            log.filter(sf.col("user_id") == "u2").select("user_id", "item_id"),
            user_features=user_features,
            item_features=item_features,
        )
        assert pred.count() == 2
        assert pred.select("user_id").distinct().collect()[0][0] == "u2"
    except:  # noqa
        pytest.fail()


def test_raises_fit(log, user_features, item_features, model):
    with pytest.raises(
        ValueError, match=r"В [\w]{4}_features отсутствуют признаки"
    ):
        model.fit(
            log.filter(sf.col("user_id") != "u1"),
            user_features.filter(sf.col("user_id") != "u2"),
            item_features,
        )


def test_raises_predict(log, user_features, item_features, model):
    with pytest.raises(ValueError, match="При обучении использовались .*"):
        model.fit(log, None, item_features)
        pred = model.predict_pairs(
            log.select("user_id", "item_id"),
            user_features=None,
            item_features=None,
        )
