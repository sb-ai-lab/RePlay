from datetime import datetime

import pandas as pd
import pytest
from pyspark.sql.types import IntegerType, StructField, StructType

from replay.data import LOG_SCHEMA, REC_SCHEMA
from replay.experimental.metrics import *


@pytest.fixture(scope="module")
def one_user():
    df = pd.DataFrame({"user_idx": [1], "item_idx": [1], "relevance": [1]})
    return df


@pytest.fixture(scope="module")
def two_users():
    df = pd.DataFrame({"user_idx": [1, 2], "item_idx": [1, 2], "relevance": [1, 1]})
    return df


@pytest.fixture(scope="module")
def recs(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, 3.0],
            [0, 1, 2.0],
            [0, 2, 1.0],
            [1, 0, 3.0],
            [1, 1, 4.0],
            [1, 4, 1.0],
            [2, 0, 5.0],
            [2, 2, 1.0],
            [2, 3, 2.0],
        ],
        schema=REC_SCHEMA,
    )


@pytest.fixture(scope="module")
def recs2(spark):
    return spark.createDataFrame(
        data=[[0, 3, 4.0], [0, 4, 5.0]],
        schema=REC_SCHEMA,
    )


@pytest.fixture(scope="module")
def empty_recs(spark):
    return spark.createDataFrame(
        data=[],
        schema=REC_SCHEMA,
    )


@pytest.fixture(scope="module")
def true(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, datetime(2019, 9, 12), 3.0],
            [0, 4, datetime(2019, 9, 13), 2.0],
            [0, 1, datetime(2019, 9, 17), 1.0],
            [1, 5, datetime(2019, 9, 14), 4.0],
            [1, 0, datetime(2019, 9, 15), 3.0],
            [2, 1, datetime(2019, 9, 15), 3.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture(scope="module")
def true_users(spark):
    return spark.createDataFrame(
        data=[[1], [2], [3], [4]],
        schema=StructType([StructField("user_idx", IntegerType())]),
    )


@pytest.fixture(scope="module")
def prev_relevance(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, 100.0],
            [0, 4, 0.0],
            [1, 10, -5.0],
            [4, 6, 11.5],
        ],
        schema=REC_SCHEMA,
    )


@pytest.fixture(scope="module")
def quality_metrics():
    return [ScalaNDCG(), ScalaHitRate(), ScalaPrecision(), ScalaRecall(), ScalaMAP(), ScalaMRR(), ScalaRocAuc()]


@pytest.fixture(scope="module")
def duplicate_recs(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, 3.0],
            [0, 1, 2.0],
            [0, 2, 1.0],
            [0, 0, 3.0],
            [1, 0, 3.0],
            [1, 1, 4.0],
            [1, 4, 1.0],
            [1, 1, 2.0],
            [2, 0, 5.0],
            [2, 2, 1.0],
            [2, 3, 2.0],
        ],
        schema=REC_SCHEMA,
    )
