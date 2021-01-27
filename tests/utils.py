# pylint: skip-file

from datetime import datetime
from typing import Dict

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from replay.constants import REC_SCHEMA, LOG_SCHEMA
from replay.session_handler import get_spark_session


def assertDictAlmostEqual(d1: Dict, d2: Dict) -> None:
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        assert_allclose(d1[key], d2[key])


@pytest.fixture
def spark():
    return get_spark_session(1, 1)


@pytest.fixture
def one_user():
    df = pd.DataFrame({"user_id": [1], "item_id": [1], "relevance": [1]})
    return df


@pytest.fixture
def two_users():
    df = pd.DataFrame(
        {"user_id": [1, 2], "item_id": [1, 2], "relevance": [1, 1]}
    )
    return df


@pytest.fixture
def recs(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", 3.0],
            ["user1", "item2", 2.0],
            ["user1", "item3", 1.0],
            ["user2", "item1", 3.0],
            ["user2", "item2", 4.0],
            ["user2", "item5", 1.0],
            ["user3", "item1", 5.0],
            ["user3", "item3", 1.0],
            ["user3", "item4", 2.0],
        ],
        schema=REC_SCHEMA,
    )


@pytest.fixture
def recs2(spark):
    return spark.createDataFrame(
        data=[["user1", "item4", 4.0], ["user1", "item5", 5.0]],
        schema=REC_SCHEMA,
    )


@pytest.fixture
def empty_recs(spark):
    return spark.createDataFrame(data=[], schema=REC_SCHEMA,)


@pytest.fixture
def true(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", datetime(2019, 9, 12), 3.0],
            ["user1", "item5", datetime(2019, 9, 13), 2.0],
            ["user1", "item2", datetime(2019, 9, 17), 1.0],
            ["user2", "item6", datetime(2019, 9, 14), 4.0],
            ["user2", "item1", datetime(2019, 9, 15), 3.0],
            ["user3", "item2", datetime(2019, 9, 15), 3.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def log2(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", datetime(2019, 9, 12), 3.0],
            ["user1", "item5", datetime(2019, 9, 13), 2.0],
            ["user1", "item2", datetime(2019, 9, 17), 1.0],
            ["user2", "item6", datetime(2019, 9, 14), 4.0],
            ["user2", "item1", datetime(2019, 9, 15), 3.0],
            ["user3", "item2", datetime(2019, 9, 15), 3.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", datetime(2019, 8, 22), 4.0],
            ["user1", "item3", datetime(2019, 8, 23), 3.0],
            ["user1", "item2", datetime(2019, 8, 27), 2.0],
            ["user2", "item4", datetime(2019, 8, 24), 3.0],
            ["user2", "item1", datetime(2019, 8, 25), 4.0],
            ["user3", "item2", datetime(2019, 8, 26), 5.0],
            ["user3", "item1", datetime(2019, 8, 26), 5.0],
            ["user3", "item3", datetime(2019, 8, 26), 3.0],
            ["user4", "item2", datetime(2019, 8, 26), 5.0],
            ["user4", "item1", datetime(2019, 8, 26), 5.0],
            ["user4", "item1", datetime(2019, 8, 26), 1.0],
        ],
        schema=LOG_SCHEMA,
    )
