import pytest

from replay.models import ThompsonSampling
from replay.utils import PYSPARK_AVAILABLE
from tests.utils import create_dataset

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf


@pytest.fixture
def preprocessed_log(log):
    return log.withColumn("relevance", sf.when(sf.col("relevance") < 3, 0).otherwise(1))


@pytest.fixture
def model():
    model = ThompsonSampling(seed=42)
    return model


@pytest.fixture
def fitted_model(preprocessed_log, model):
    dataset = create_dataset(preprocessed_log)
    model.fit(dataset)
    return model


@pytest.mark.spark
def test_works(preprocessed_log, model):
    dataset = create_dataset(preprocessed_log)
    model.fit(dataset)
    model.item_popularity.count()


@pytest.mark.core
def test_tsampling_init_args(model):
    assert model._init_args["seed"] == 42


@pytest.mark.spark
@pytest.mark.parametrize(
    "sample,seed",
    [(False, None), (True, None)],
    ids=[
        "no_sampling",
        "sample_not_fixed",
    ],
)
def test_predict_empty_log(fitted_model, preprocessed_log, sample, seed):
    fitted_model.seed = seed
    fitted_model.sample = sample

    users = preprocessed_log.select("user_idx").distinct()
    pred = fitted_model.predict(dataset=None, queries=users, items=list(range(10)), k=1)
    assert pred.count() == users.count()


@pytest.mark.spark
def test_predict(preprocessed_log, model):
    dataset = create_dataset(preprocessed_log)
    model.fit(dataset)
    recs = model.predict(dataset, k=1, queries=[1, 0], items=[3, 2])
    assert recs.count() == 2
    assert recs.select(sf.sum(sf.col("user_idx").isin([1, 0]).astype("int"))).collect()[0][0] == 2
