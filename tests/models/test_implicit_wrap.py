import pandas as pd
import implicit
import pytest
import numpy as np
from datetime import datetime
from pyspark.sql.types import (
    IntegerType,
    StructField,
    StructType,
)

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


@pytest.fixture
def pairs(spark):
    return spark.createDataFrame(
        data=[
            [1, 1],
            [2, 1],
        ],
        schema=StructType([
            StructField("user_idx", IntegerType()),
            StructField("item_idx", IntegerType()),
        ])
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


@pytest.mark.parametrize(
    "model",
    [
        ImplicitWrap(implicit.als.AlternatingLeastSquares()),
        ImplicitWrap(implicit.bpr.BayesianPersonalizedRanking()),
        ImplicitWrap(implicit.lmf.LogisticMatrixFactorization()),
    ]
)
def test_predict_pairs(model, log, pairs):
    model.fit(log)
    pred_no_log = model.predict_pairs(pairs)
    pred_log = model.predict_pairs(pairs, log)

    assert len(pred_log.toPandas()["user_idx"].unique()) == len(pred_no_log.toPandas()["user_idx"].unique()) == 2
    assert np.allclose(pairs.toPandas()["user_idx"], pred_no_log.toPandas()["user_idx"], pred_log.toPandas()["user_idx"])
    assert np.allclose(pairs.toPandas()["item_idx"], pred_no_log.toPandas()["item_idx"], pred_log.toPandas()["item_idx"])
