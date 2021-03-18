# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from replay.constants import LOG_SCHEMA
from replay.models import ALSWrap
from tests.utils import log


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
