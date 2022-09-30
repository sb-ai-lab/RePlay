import pandas as pd
import implicit
import pytest
from datetime import datetime

from replay.constants import LOG_SCHEMA
from replay.models import ImplicitWrap
from tests.utils import spark
from replay.session_handler import get_spark_session


@pytest.fixture
def spark():
    return get_spark_session(1, 1)


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
    assert len(pred.toPandas()["user_idx"].unique()) == 1
    assert pred.toPandas().shape[0] == 2
