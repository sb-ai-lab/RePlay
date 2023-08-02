# pylint: disable-all
from datetime import datetime

import pytest
from pyspark.sql import functions as sf

from replay.data import LOG_SCHEMA
from replay.models import PopRec
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
def model():
    model = PopRec()
    return model


def test_works(log, model):
    try:
        pred = model.fit_predict(log, k=1)
        assert list(pred.toPandas().sort_values("user_idx")["item_idx"]) == [
            1,
            2,
            0,
        ]
    except:  # noqa
        pytest.fail()


def test_clear_cache(log, model):
    try:
        model.fit(log)
        model._clear_cache()
    except:  # noqa
        pytest.fail()
