# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np
from pyspark.sql import functions as sf

from replay.constants import LOG_SCHEMA
from replay.models import SLIM
from tests.utils import spark, sparkDataFrameEqual


@pytest.fixture
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            ["u1", "i1", date, 1.0],
            ["u2", "i1", date, 1.0],
            ["u3", "i3", date, 2.0],
            ["u2", "i3", date, 2.0],
            ["u3", "i4", date, 2.0],
            ["u1", "i4", date, 2.0],
            ["u4", "i1", date, 2.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def model():
    return SLIM(0.0, 0.01, seed=42)


def test_fit(log, model):
    model.fit(log)
    assert np.allclose(
        model.similarity.toPandas().sort_values("item_id_one").to_numpy(),
        [
            (0, 1, 0.163338303565979),
            (0, 2, 0.1633233278989792),
            (1, 0, 0.17635512351989746),
            (1, 2, 0.45091119408607483),
            (2, 0, 0.17635512351989746),
            (2, 1, 0.45091116428375244),
        ],
    )


def test_predict(log, model):
    model.fit(log)
    recs = model.predict(log, k=1)
    assert np.allclose(
        recs.toPandas().sort_values("user_id", ascending=False).relevance,
        [
            0.163338303565979,
            0.3527102470397949,
            0.614234521985054,
            0.6142494678497314,
        ],
    )


def test_predict_pairs(log, model):
    try:
        model.fit(log)
        pred = model.predict_pairs(log.filter(sf.col("user_id") == "u2"), log)
        assert pred.count() == 2
        sparkDataFrameEqual(
            log.filter(sf.col("user_id") == "u2").select("user_id", "item_id"),
            pred.select("user_id", "item_id"),
        )
    except:  # noqa
        pytest.fail()


def test_predict_pairs_raises(log, model):
    with pytest.raises(ValueError):
        model.fit(log)
        model.predict_pairs(log.filter(sf.col("user_id") == "u1"))


@pytest.mark.parametrize(
    "beta,lambda_", [(0.0, 0.0), (-0.1, 0.1), (0.1, -0.1)]
)
def test_exceptions(beta, lambda_):
    with pytest.raises(ValueError):
        SLIM(beta, lambda_)
