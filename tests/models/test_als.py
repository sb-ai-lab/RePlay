# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.models import ALSWrap
from tests.utils import log, spark


@pytest.fixture
def model():
    model = ALSWrap(1, implicit_prefs=False)
    model._seed = 42
    return model


def test_works(log, model):
    try:
        pred = model.fit_predict(log, k=1)
        assert pred.count() == 4
    except:  # noqa
        pytest.fail()


def test_diff_feedback_type(log, model):
    pred_exp = model.fit_predict(log, k=1)
    model.implicit_prefs = True
    pred_imp = model.fit_predict(log, k=1)
    assert not np.allclose(
        pred_exp.toPandas().sort_values("user_id")["relevance"].values,
        pred_imp.toPandas().sort_values("user_id")["relevance"].values,
    )
