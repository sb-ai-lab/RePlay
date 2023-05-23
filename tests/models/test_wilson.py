# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest
import numpy as np

from pyspark.sql import functions as sf
from statsmodels.stats.proportion import proportion_confint

from replay.models import Wilson
from replay.utils import convert2spark
from tests.utils import log, pos_neg_log, spark, sparkDataFrameEqual


@pytest.fixture
def model():
    model = Wilson()
    return model


def test_works(pos_neg_log, model):
    model.fit(pos_neg_log)
    model.item_popularity.count()


def calc_wilson_interval(pos_neg_log):
    data_frame = (
        pos_neg_log.groupby("item_idx")
        .agg(
            sf.sum("relevance").alias("pos"),
            sf.count("relevance").alias("total"),
        )
        .toPandas()
    )
    pos = np.array(data_frame["pos"].values)
    total = np.array(data_frame["total"].values)
    data_frame["relevance"] = proportion_confint(pos, total, method="wilson")[
        0
    ]
    data_frame = data_frame.drop(["pos", "total"], axis=1)
    return convert2spark(data_frame)


def test_calculation(pos_neg_log, model):
    model.fit(pos_neg_log)
    stat_wilson = calc_wilson_interval(pos_neg_log)
    sparkDataFrameEqual(model.item_popularity, stat_wilson)


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
