# pylint: disable-all
from datetime import datetime

import pytest

from replay.constants import LOG_SCHEMA
from replay.models import KNN
from tests.utils import spark


@pytest.fixture
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            ["u1", "i1", date, 1.0],
            ["u2", "i2", date, 1.0],
            ["u3", "i1", date, 1.0],
            ["u3", "i2", date, 1.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def model():
    model = KNN(1)
    return model


def test_works(log, model):
    model.fit(log)
    recs = model.predict(log, k=1, users=["u1", "u2"]).toPandas()
    assert recs.loc[recs["user_id"] == "u1", "item_id"].iloc[0] == "i2"
    assert recs.loc[recs["user_id"] == "u2", "item_id"].iloc[0] == "i1"
