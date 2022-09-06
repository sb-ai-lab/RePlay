# pylint: disable-all
from datetime import datetime

import pytest
from pyspark.sql import DataFrame

from replay.constants import LOG_SCHEMA
from replay.models import CQL
from tests.utils import spark


@pytest.fixture
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, 0, date, 1.0],
            [1, 1, date, 1.0],
            [2, 0, date, 1.0],
            [2, 1, date, 1.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def model():
    model = CQL(k=1, n_epochs=1)
    return model


def test_works(log: DataFrame, model: CQL):
    model.fit(log)
    recs = model.predict(log, k=1, users=[0, 1]).toPandas()
    assert recs.loc[recs["user_idx"] == 0, "item_idx"].iloc[0] == 1
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] == 0

