# pylint: disable-all
from datetime import datetime

import pytest
from pyspark.sql import functions as sf

from replay.constants import LOG_SCHEMA
from replay.models import PopRec
from tests.utils import log, spark


@pytest.fixture
def model():
    model = PopRec()
    return model


def test_works(log, model):
    try:
        pred = model.fit_predict(log, k=1)
        assert list(pred.toPandas().sort_values("user_idx")["item_idx"]) == [
            3,
            1,
            3,
            2,
        ]
    except:  # noqa
        pytest.fail()


def test_clear_cache(log, model):
    try:
        model.fit(log)
        model._clear_cache()
    except:  # noqa
        pytest.fail()
