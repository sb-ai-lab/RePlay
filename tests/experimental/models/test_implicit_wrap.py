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

from replay.data import INTERACTIONS_SCHEMA
from replay.experimental.models import ImplicitWrap
from tests.utils import spark, log, sparkDataFrameEqual, long_log_with_features
from implicit.als import AlternatingLeastSquares
from replay.utils.session_handler import get_spark_session


@pytest.mark.experimental
@pytest.mark.parametrize(
    "model",
    [
        ImplicitWrap(implicit.als.AlternatingLeastSquares()),
        ImplicitWrap(implicit.bpr.BayesianPersonalizedRanking()),
        ImplicitWrap(implicit.lmf.LogisticMatrixFactorization()),
    ]
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


@pytest.mark.experimental
@pytest.mark.parametrize(
    "model",
    [
        ImplicitWrap(implicit.als.AlternatingLeastSquares()),
        ImplicitWrap(implicit.bpr.BayesianPersonalizedRanking()),
        ImplicitWrap(implicit.lmf.LogisticMatrixFactorization()),
    ]
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
    pred_top_k = model.predict_pairs(pairs, log if log_in_pred else None, k=2)

    assert pred.select("user_idx").distinct().count() == 1
    assert pred_top_k.groupBy("user_idx").count().filter(f"count > 2").count() == 0
    assert pred.groupBy("user_idx").count().filter(f"count > 2").count() != 0

    sparkDataFrameEqual(pairs.select("user_idx","item_idx"), pred.select("user_idx","item_idx"))


@pytest.mark.experimental
@pytest.mark.parametrize(
    "model",
    [
        ImplicitWrap(implicit.als.AlternatingLeastSquares()),
        ImplicitWrap(implicit.bpr.BayesianPersonalizedRanking()),
        ImplicitWrap(implicit.lmf.LogisticMatrixFactorization()),
    ]
)
def test_predict_empty_log(log, model):
    model.fit(log)
    model.predict(log.limit(0), 1)
