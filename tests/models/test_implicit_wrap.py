import pandas as pd
import implicit
import pytest
from datetime import datetime

from replay.constants import LOG_SCHEMA
from replay.models import ImplicitWrap
from tests.utils import spark, log
from replay.session_handler import get_spark_session


@pytest.mark.parametrize(
    "model",
    [
        ImplicitWrap(implicit.als.AlternatingLeastSquares()),
        ImplicitWrap(implicit.bpr.BayesianPersonalizedRanking()),
        ImplicitWrap(implicit.lmf.LogisticMatrixFactorization()),
    ]
)
def test_predict(model, log):
    model.fit(log)
    pred = model.predict(
        log=log,
        k=2,
        users=[1],
        filter_seen_items=False
    )
    assert pred.select("user_idx").distinct().count() == 1
    assert pred.count() == 2
