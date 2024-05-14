from datetime import datetime

import pytest

from replay.data import get_schema
from replay.models import PopRec
from tests.utils import create_dataset

pyspark = pytest.importorskip("pyspark")

INTERACTIONS_SCHEMA = get_schema("user_idx", "item_idx", "timestamp", "relevance")


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
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture
def model():
    model = PopRec()
    return model


@pytest.mark.spark
def test_works(log, model):
    dataset = create_dataset(log)
    pred = model.fit_predict(dataset, k=1)
    assert list(pred.toPandas().sort_values("user_idx")["item_idx"]) == [1, 2, 0]


@pytest.mark.spark
def test_clear_cache(log, model):
    dataset = create_dataset(log)
    model.fit(dataset)
    model._clear_cache()
