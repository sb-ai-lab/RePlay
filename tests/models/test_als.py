# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from replay.constants import LOG_SCHEMA
from replay.models import ALSWrap
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
    model = ALSWrap(1)
    model._seed = 42
    return model


def test_works(log, model):
    try:
        model.fit_predict(log, k=1)
    except:  # noqa
        pytest.fail()
