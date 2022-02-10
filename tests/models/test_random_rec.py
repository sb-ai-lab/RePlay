# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest

from pyspark.sql import functions as sf

from replay.models import RandomRec
from tests.utils import log, spark, sparkDataFrameEqual, sparkDataFrameNotEqual


@pytest.fixture(
    params=[{"seed": 123}, {}, {"distribution": "popular_based", "seed": 123}],
    ids=["uniform_seed", "uniform_no_seed", "popular_based_seed"],
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
            .withColumn("probability", sf.lit(1.0))
        )
    else:
        true_matrix = log.groupby(
            "item_idx"
        ).agg(  # pylint: disable=not-callable
            sf.countDistinct("user_idx").astype("double").alias("probability")
        )

    sparkDataFrameEqual(
        fitted_model.item_popularity, true_matrix,
    )


def test_predict(fitted_model, log):
    # fixed seed provides reproducibility, non-fixed provides diversity
    equality_check = (
        sparkDataFrameNotEqual
        if fitted_model.seed is None
        else sparkDataFrameEqual
    )
    pred = fitted_model.predict(log, k=1)
    fitted_model.fit(log)
    pred_after_refit = fitted_model.predict(log, k=1)
    equality_check(pred, pred_after_refit)
    pred_repeat = fitted_model.predict(log, k=1)
    equality_check(pred_after_refit, pred_repeat)
