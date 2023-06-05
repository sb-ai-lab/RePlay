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
from replay.model_handler import save, load
from replay.models import ImplicitWrap
from tests.utils import spark, log, sparkDataFrameEqual
from replay.session_handler import get_spark_session


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
