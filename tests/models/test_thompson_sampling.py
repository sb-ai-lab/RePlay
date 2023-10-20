# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.models import ThompsonSampling
from replay.models import UCB
from tests.utils import log, spark, sparkDataFrameEqual


@pytest.fixture
def preprocessed_log(log):
    return log.withColumn(
        "relevance", sf.when(sf.col("relevance") < 3, 0).otherwise(1)
    )


@pytest.fixture
def model():
    model = ThompsonSampling(seed=42)
    return model


@pytest.fixture
def fitted_model(preprocessed_log, model):
    model.fit(preprocessed_log)
    return model


def test_works(preprocessed_log, model):
    model.fit(preprocessed_log)
    model.item_popularity.count()


def test_tsampling_init_args(model):
    assert model._init_args["seed"] == 42


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
    pred = fitted_model.predict(
        log=None, users=users, items=list(range(10)), k=1
    )
    assert pred.count() == users.count()


def test_predict(preprocessed_log, model):
    model.fit(preprocessed_log)
    recs = model.predict(preprocessed_log, k=1, users=[1, 0], items=[3, 2])
    assert recs.count() == 2
    assert (
        recs.select(
            sf.sum(sf.col("user_idx").isin([1, 0]).astype("int"))
        ).collect()[0][0]
        == 2
    )
