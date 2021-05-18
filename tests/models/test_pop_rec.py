# pylint: disable-all
from datetime import datetime

import pytest
from pyspark.sql import functions as sf

from replay.constants import LOG_SCHEMA
from replay.models import PopRec
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
def model():
    model = PopRec()
    return model


def test_works(log, model):
    try:
        pred = model.fit_predict(log, k=1)
        assert list(pred.toPandas().sort_values("user_id")["item_id"]) == [
            "i3",
            "i4",
            "i1",
        ]
    except:  # noqa
        pytest.fail()


def test_predict_pairs(log, model):
    try:
        model.fit(log.filter(sf.col("item_id") != "i4"))
        # исходное количество пар - 2
        pred = model.predict_pairs(log.filter(sf.col("user_id") == "u1"))
        # для холодного объекта не возвращаем ничего
        assert pred.count() == 1
        assert pred.select("user_id").distinct().collect()[0][0] == "u1"
        # предсказываем для всех теплых объектов
        assert pred.select("item_id").distinct().collect()[0][0] == "i1"
    except:  # noqa
        pytest.fail()


def test_clear_cache(model):
    try:
        model._clear_cache()
    except:  # noqa
        pytest.fail()
