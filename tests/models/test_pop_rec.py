from datetime import datetime

import pytest

from replay.data import get_schema
from replay.models import PopRec
from tests.utils import create_dataset

pyspark = pytest.importorskip("pyspark")

INTERACTIONS_SCHEMA = get_schema("user_idx", "item_idx", "timestamp", "relevance")


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, type_of_impl",
    [(PopRec, "polars"), (PopRec, "pandas"), (PopRec, "spark")],
    ids=["pop_rec_polars", "pop_rec_pandas", "pop_rec_spark"],
)
def test_use_rating_invalid(base_model, type_of_impl, datasets):
    model = base_model()
    assert not model.is_fitted
    assert model.use_rating is False
    with pytest.raises(ValueError, match="incorrect type of argument 'value'"):
        model.use_rating = 2


@pytest.mark.spark
@pytest.mark.parametrize(
    "base_model, type_of_impl",
    [(PopRec, "polars"), (PopRec, "pandas"), (PopRec, "spark")],
    ids=["pop_rec_polars", "pop_rec_pandas", "pop_rec_spark"],
)
def test_use_rating_valid(base_model, type_of_impl, datasets):
    model = base_model()
    assert not model.is_fitted
    assert model.use_rating is False
    model.use_rating = True
    assert model._init_when_first_impl_arrived_args["use_rating"] is True
    model.fit(datasets[type_of_impl])
    assert model.is_fitted
    assert model.use_rating is True
