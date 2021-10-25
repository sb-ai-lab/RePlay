# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest
import numpy as np

from pyspark.sql import functions as sf
from statsmodels.stats.proportion import proportion_confint

from replay.models import Wilson
from replay.utils import convert2spark
from tests.utils import log, spark, sparkDataFrameEqual


@pytest.fixture
def model():
    model = Wilson()
    return model


def test_works(log, model):
    log = log.withColumn(
        "relevance", sf.when(sf.col("relevance") < 3, 0).otherwise(1)
    )
    model.fit(log)
    model.item_popularity.count()


def calc_wilson_interval(log):
    data_frame = (
        log.groupby("item_id")
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


def test_calculation(log, model):
    log = log.withColumn(
        "relevance", sf.when(sf.col("relevance") < 3, 0).otherwise(1)
    )
    model.fit(log)
    stat_wilson = calc_wilson_interval(log)
    spark_wilson = model._convert_back(
        model.item_popularity,
        log.schema["user_id"].dataType,
        log.schema["item_id"].dataType,
    )
    sparkDataFrameEqual(spark_wilson, stat_wilson)


def test_predict(log, model):
    log = log.withColumn(
        "relevance", sf.when(sf.col("relevance") < 3, 0).otherwise(1)
    )
    model.fit(log)
    recs = model.predict(
        log, k=1, users=["user2", "user1"], items=["item4", "item3"]
    )
    assert recs.count() == 2
    assert (
        recs.select(
            sf.sum(sf.col("user_id").isin(["user2", "user1"]).astype("int"))
        ).collect()[0][0]
        == 2
    )
