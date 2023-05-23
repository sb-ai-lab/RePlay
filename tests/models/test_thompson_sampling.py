# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.models import ThompsonSampling
from replay.models import UCB
from tests.utils import log, pos_neg_log, spark, sparkDataFrameEqual


@pytest.fixture
def model():
    model = ThompsonSampling(seed=42)
    return model


@pytest.fixture
def fitted_model(pos_neg_log, model):
    model.fit(pos_neg_log)
    return model


def test_works(pos_neg_log, model):
    model.fit(pos_neg_log)
    model.item_popularity.count()


@pytest.mark.parametrize(
    "sample,seed",
    [(False, None), (True, None)],
    ids=[
        "no_sampling",
        "sample_not_fixed",
    ],
)
def test_predict_empty_log(fitted_model, pos_neg_log, sample, seed):
    fitted_model.seed = seed
    fitted_model.sample = sample

    users = pos_neg_log.select("user_idx").distinct()
    pred = fitted_model.predict(
        log=None, users=users, items=list(range(10)), k=1
    )
    assert pred.count() == users.count()


def test_predict(pos_neg_log, model):
    model.fit(pos_neg_log)
    recs = model.predict(pos_neg_log, k=1, users=[1, 0], items=[3, 2])
    assert recs.count() == 2
    assert (
        recs.select(
            sf.sum(sf.col("user_idx").isin([1, 0]).astype("int"))
        ).collect()[0][0]
        == 2
    )
