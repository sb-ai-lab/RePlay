# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

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
    return spark.createDataFrame([("u1", 2.0, 3.0)]).toDF(
        "user_id", "user_feature_1", "user_feature_2"
    )


@pytest.fixture
def item_features(spark):
    return spark.createDataFrame([("i1", 4.0, 5.0)]).toDF(
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
        users=user_features.select("user_id"),
        items=item_features.select("item_id"),
        user_features=user_features,
        item_features=item_features,
        filter_seen_items=True,
    )
    assert pred.count() == 3
