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
from tests.utils import spark, log, sparkDataFrameEqual
from replay.session_handler import get_spark_session


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
    assert pred.select("user_idx").distinct().count() == 1
    assert pred.count() == 2


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

    assert pred_log.select("user_idx").distinct().count() == pred_no_log.select("user_idx").distinct().count() == 2
    sparkDataFrameEqual(pairs.select("user_idx","item_idx"), pred_no_log.select("user_idx","item_idx"))
    sparkDataFrameEqual(pred_log.select("user_idx","item_idx"), pred_no_log.select("user_idx","item_idx"))
