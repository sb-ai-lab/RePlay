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
            [0, 0, date, 1.0],
            [1, 0, date, 1.0],
            [2, 1, date, 2.0],
            [2, 1, date, 2.0],
            [1, 1, date, 2.0],
            [2, 3, date, 2.0],
            [0, 3, date, 2.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def model():
    return Word2VecRec(
        rank=1, window_size=1, use_idf=True, seed=42, min_count=0
    )


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
    print(vectors)
    assert np.allclose(
        vectors,
        [[1, 5.33072205e-04], [0, 1.54904364e-01], [3, 2.13002899e-01]],
    )


def test_predict(log, model):
    model.fit(log)
    recs = model.predict(log, k=1)
    recs.show()
    assert np.allclose(
        recs.toPandas().sort_values("user_idx").relevance,
        [1.0003180271011836, 0.9653348251181987, 0.972993367280087],
    )
