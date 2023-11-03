# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest

from pyspark.sql import functions as sf

from replay.experimental.models import RandomRec
from tests.utils import log, spark, sparkDataFrameEqual, sparkDataFrameNotEqual


@pytest.fixture(
    params=[
        {"seed": 123},
        {},
        {"distribution": "popular_based", "seed": 123},
        {"distribution": "relevance", "seed": 123},
    ],
    ids=[
        "uniform_seed",
        "uniform_no_seed",
        "popular_based_seed",
        "relevance_seed",
    ],
)
def fitted_model(request, log):
    model = RandomRec(**request.param)
    model.fit(log)
    return model


def test_popularity_matrix(log, fitted_model):
    if fitted_model.distribution == "uniform":
        true_matrix = (
            log.select("item_idx")
            .distinct()
            .withColumn("relevance", sf.lit(1.0))
        )
    elif fitted_model.distribution == "popular_based":
        true_matrix = log.groupby(
            "item_idx"
        ).agg(  # pylint: disable=not-callable
            sf.countDistinct("user_idx").astype("double").alias("relevance")
        )
    elif fitted_model.distribution == "relevance":
        true_matrix = log.groupby("item_idx").agg(
            sf.sum("relevance").alias("relevance")
        )

    true_matrix = true_matrix.withColumn(
        "relevance",
        sf.col("relevance")
        / sf.lit(true_matrix.agg(sf.sum("relevance")).first()[0]),
    )

    sparkDataFrameEqual(
        fitted_model.item_popularity.orderBy(["item_idx"]),
        true_matrix.orderBy(["item_idx"]),
    )


def test_predict(fitted_model, log):
    # fixed seed provides reproducibility (the same prediction every time),
    # non-fixed provides diversity (predictions differ every time)
    equality_check = (
        sparkDataFrameNotEqual
        if fitted_model.seed is None
        else sparkDataFrameEqual
    )
    pred = fitted_model.predict(log, k=1)
    pred_checkpoint = pred.localCheckpoint()
    pred.unpersist()

    # predictions are equal/non-equal after model re-fit
    fitted_model.fit(log)
    pred_after_refit = fitted_model.predict(log, k=1)
    equality_check(pred_checkpoint, pred_after_refit)

    # predictions are equal/non-equal when call `predict repeatedly`
    pred_after_refit_checkpoint = pred_after_refit.localCheckpoint()
    pred_after_refit.unpersist()
    pred_repeat = fitted_model.predict(log, k=1)
    equality_check(pred_after_refit_checkpoint, pred_repeat)


def test_predict_to_file(spark, fitted_model, log, tmp_path):
    path = str((tmp_path / "pred.parquet").resolve().absolute())
    fitted_model.predict(log, k=10, recs_file_path=path)
    pred_cached = fitted_model.predict(log, k=10, recs_file_path=None)
    pred_from_file = spark.read.parquet(path)
    if fitted_model.seed is not None:
        sparkDataFrameEqual(pred_cached, pred_from_file)
    else:
        sparkDataFrameNotEqual(pred_cached, pred_from_file)
