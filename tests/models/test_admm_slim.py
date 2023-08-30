# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np
from pyspark.sql import functions as sf

from replay.data import LOG_SCHEMA
from replay.models import ADMMSLIM
from tests.utils import spark


@pytest.fixture
def log(spark):
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
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def model():
    return ADMMSLIM(1, 10, 42)


def test_fit(log, model):
    model.fit(log)
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


def test_predict(log, model):
    model.fit(log)
    recs = model.predict(log, k=1)
    assert recs.count() == 4


@pytest.mark.parametrize(
    "lambda_1,lambda_2", [(0.0, 0.0), (-0.1, 0.1), (0.1, -0.1)]
)
def test_exceptions(lambda_1, lambda_2):
    with pytest.raises(ValueError):
        ADMMSLIM(lambda_1, lambda_2)
