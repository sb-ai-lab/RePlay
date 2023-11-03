# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest

from pyspark.sql import functions as sf

from replay.models import UCB
from tests.utils import log, log2, spark, sparkDataFrameEqual, sparkDataFrameNotEqual, create_dataset


@pytest.fixture
def log_ucb(log):
    return log.withColumn(
        "relevance", sf.when(sf.col("relevance") > 3, 1).otherwise(0)
    )


@pytest.fixture
def log_ucb2(log2):
    return log2.withColumn(
        "relevance", sf.when(sf.col("relevance") > 3, 1).otherwise(0)
    )


@pytest.fixture
def fitted_model(log_ucb):
    model = UCB()
    dataset = create_dataset(log_ucb)
    model.fit(dataset)
    return model


def test_popularity_matrix(fitted_model, log_ucb):
    assert (
        fitted_model.item_popularity.count()
        == log_ucb.select("item_idx").distinct().count()
    )
    fitted_model.item_popularity.show()


# @pytest.mark.parametrize(
#     "sample,seed",
#     [(False, None), (True, None)],
#     ids=[
#         "no_sampling",
#         "sample_not_fixed",
#     ],
# )
# def test_predict_empty_log(fitted_model, log_ucb, sample, seed):
#     fitted_model.seed = seed
#     fitted_model.sample = sample

#     queries = log_ucb.select("user_idx").distinct()
#     pred = fitted_model.predict(
#         dataset=None, queries=queries, items=list(range(10)), k=1
#     )
#     assert pred.count() == queries.count()


@pytest.mark.parametrize(
    "sample,seed",
    [(False, None), (True, None), (True, 123)],
    ids=[
        "no_sampling",
        "sample_not_fixed",
        "sample_fixed",
    ],
)
def test_predict(fitted_model, log_ucb, sample, seed):
    # fixed seed provides reproducibility (the same prediction every time),
    # non-fixed provides diversity (predictions differ every time)
    fitted_model.seed = seed
    # fitted_model.sample = sample

    equality_check = (
        sparkDataFrameNotEqual
        if fitted_model.sample and fitted_model.seed is None
        else sparkDataFrameEqual
    )

    # add more items to get more randomness
    dataset = create_dataset(log_ucb)
    pred = fitted_model.predict(dataset, items=list(range(10)), k=1)
    pred_checkpoint = pred.localCheckpoint()
    pred.unpersist()

    # predictions are equal/non-equal after model re-fit
    fitted_model.fit(dataset)

    pred_after_refit = fitted_model.predict(
        dataset, items=list(range(10)), k=1
    )
    equality_check(pred_checkpoint, pred_after_refit)

    # predictions are equal/non-equal when call `predict repeatedly`
    pred_after_refit_checkpoint = pred_after_refit.localCheckpoint()
    pred_after_refit.unpersist()
    pred_repeat = fitted_model.predict(dataset, items=list(range(10)), k=1)
    equality_check(pred_after_refit_checkpoint, pred_repeat)


def test_refit(fitted_model, log_ucb, log_ucb2):

    fitted_model.seed = 123
    # fitted_model.sample = True

    equality_check = (
        sparkDataFrameNotEqual
        if fitted_model.sample and fitted_model.seed is None
        else sparkDataFrameEqual
    )
    dataset = create_dataset(log_ucb)
    dataset2 = create_dataset(log_ucb2)
    fitted_model.refit(dataset2)
    pred_after_refit = fitted_model.predict(
        dataset, items=list(range(10)), k=1
    )

    united_dataset = create_dataset(log_ucb.union(log_ucb2))
    fitted_model.fit(united_dataset)
    pred_after_full_fit = fitted_model.predict(
        dataset, items=list(range(10)), k=1
    )

    # predictions are equal/non-equal after model refit and full fit on all log
    equality_check(pred_after_full_fit, pred_after_refit)
