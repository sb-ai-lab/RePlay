# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from replay.constants import LOG_SCHEMA
from replay.models import ALSWrap
from tests.utils import log, spark


@pytest.fixture
def model():
    model = ALSWrap(1)
    model._seed = 42
    return model


def test_works(log, model):
    try:
        pred = model.fit_predict(log, k=1)
        np.allclose(
            pred.toPandas().sort_values("user_id")["relevance"].values,
            [0.559088, 0.816796, 0.566497, 0.783326],
        )
    except:  # noqa
        pytest.fail()
