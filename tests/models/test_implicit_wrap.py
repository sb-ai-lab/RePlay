import pandas as pd
import implicit
import pytest
import numpy as np
from datetime import datetime
from pyspark.sql import functions as sf
from pyspark.sql.types import (
    IntegerType,
    StructField,
    StructType,
)

from replay.constants import LOG_SCHEMA
from replay.models import ImplicitWrap
from tests.utils import spark, log, sparkDataFrameEqual
from replay.session_handler import get_spark_session


test_models = [
    ImplicitWrap(implicit.als.AlternatingLeastSquares()),
    ImplicitWrap(implicit.bpr.BayesianPersonalizedRanking()),
    ImplicitWrap(implicit.lmf.LogisticMatrixFactorization()),
]


@pytest.mark.parametrize(
    "model",
    test_models
)
@pytest.mark.parametrize(
    "filter_seen",
    [
        True,
        False
    ]
)
def test_predict(model, log,filter_seen):
    model.fit(log)
    pred = model.predict(
        log=log,
        k=5,
        users=[1],
        filter_seen_items=filter_seen
    )

    assert pred.select("user_idx").distinct().count() == 1
    assert pred.count() == 2 if filter_seen else 4


@pytest.mark.parametrize(
    "model",
    test_models
)
@pytest.mark.parametrize(
    "log_in_pred",
    [
        True,
        False
    ]
)
def test_predict_pairs(model, log, log_in_pred):
    pairs = log.select("user_idx","item_idx").filter(sf.col("user_idx") == 2)
    model.fit(log)
    pred = model.predict_pairs(pairs, log if log_in_pred else None)

    assert pred.select("user_idx").distinct().count() == 1
    sparkDataFrameEqual(pairs.select("user_idx","item_idx"), pred.select("user_idx","item_idx"))
