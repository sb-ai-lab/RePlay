# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import numpy as np
import pytest
from statsmodels.stats.proportion import proportion_confint

from replay.models import Wilson
from tests.utils import create_dataset, log, log2, spark, sparkDataFrameEqual, sparkDataFrameNotEqual

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import functions as sf

from replay.utils.spark_utils import convert2spark


@pytest.fixture
def model():
    model = Wilson()
    return model


@pytest.mark.spark
def test_works(log, model):
    log = log.withColumn(
        "relevance", sf.when(sf.col("relevance") < 3, 0).otherwise(1)
    )
    dataset = create_dataset(log)
    model.fit(dataset)
    model.item_popularity.count()


def calc_wilson_interval(log):
    data_frame = (
        log.groupby("item_idx")
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


@pytest.mark.spark
def test_calculation(log, model):
    log = log.withColumn(
        "relevance", sf.when(sf.col("relevance") < 3, 0).otherwise(1)
    )
    dataset = create_dataset(log)
    model.fit(dataset)
    stat_wilson = calc_wilson_interval(log)
    sparkDataFrameEqual(model.item_popularity, stat_wilson)


@pytest.mark.spark
def test_predict(log, model):
    log = log.withColumn(
        "relevance", sf.when(sf.col("relevance") < 3, 0).otherwise(1)
    )
    dataset = create_dataset(log)
    model.fit(dataset)
    recs = model.predict(dataset, k=1, queries=[1, 0], items=[3, 2])
    assert recs.count() == 2
    assert (
        recs.select(
            sf.sum(sf.col("user_idx").isin([1, 0]).astype("int"))
        ).collect()[0][0]
        == 2
    )
