# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np
from pyspark.sql import functions as sf

from replay.constants import LOG_SCHEMA
from replay.models import Word2VecRec
from replay.utils import vector_dot
from tests.utils import spark


@pytest.fixture
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            ["u1", "i1", date, 1.0],
            ["u2", "i1", date, 1.0],
            ["u3", "i3", date, 2.0],
            ["u3", "i3", date, 2.0],
            ["u2", "i3", date, 2.0],
            ["u3", "i4", date, 2.0],
            ["u1", "i4", date, 2.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def model():
    return Word2VecRec(rank=1, window_size=1, use_idf=True, seed=42)


def test_fit(log, model):
    model.fit(log)
    vectors = (
        model.vectors.select(
            "item",
            vector_dot(sf.col("vector"), sf.col("vector")).alias("norm"),
        )
        .toPandas()
        .to_numpy()
    )
    assert np.allclose(
        vectors,
        [[0, 5.45887464e-04], [2, 1.54838404e-01], [1, 2.13055389e-01],],
    )


def test_predict(log, model):
    model.fit(log)
    recs = model.predict(log, k=1)
    assert np.allclose(
        recs.toPandas().sort_values("user_id").relevance,
        [1.000322493440465, 0.9613139892286415, 0.9783670469059589,],
    )
