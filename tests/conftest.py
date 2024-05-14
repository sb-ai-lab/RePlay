from datetime import datetime

import pytest

from replay.data import get_schema
from replay.utils import PYSPARK_AVAILABLE
from replay.utils.session_handler import get_spark_session

if PYSPARK_AVAILABLE:
    INTERACTIONS_SCHEMA = get_schema("user_idx", "item_idx", "timestamp", "relevance")


@pytest.fixture
def spark():
    session = get_spark_session()
    session.sparkContext.setLogLevel("ERROR")
    return session


@pytest.fixture
def log_to_pred(spark):
    return spark.createDataFrame(
        data=[
            [0, 2, datetime(2019, 9, 12), 3.0],
            [0, 4, datetime(2019, 9, 13), 2.0],
            [1, 5, datetime(2019, 9, 14), 4.0],
            [4, 0, datetime(2019, 9, 15), 3.0],
            [4, 1, datetime(2019, 9, 15), 3.0],
        ],
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture
def log2(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, datetime(2019, 9, 12), 3.0],
            [0, 2, datetime(2019, 9, 13), 2.0],
            [0, 1, datetime(2019, 9, 17), 1.0],
            [1, 3, datetime(2019, 9, 14), 4.0],
            [1, 0, datetime(2019, 9, 15), 3.0],
            [2, 1, datetime(2019, 9, 15), 3.0],
        ],
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, datetime(2019, 8, 22), 4.0],
            [0, 2, datetime(2019, 8, 23), 3.0],
            [0, 1, datetime(2019, 8, 27), 2.0],
            [1, 3, datetime(2019, 8, 24), 3.0],
            [1, 0, datetime(2019, 8, 25), 4.0],
            [2, 1, datetime(2019, 8, 26), 5.0],
            [2, 0, datetime(2019, 8, 26), 5.0],
            [2, 2, datetime(2019, 8, 26), 3.0],
            [3, 1, datetime(2019, 8, 26), 5.0],
            [3, 0, datetime(2019, 8, 26), 5.0],
            [3, 0, datetime(2019, 8, 26), 1.0],
        ],
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture
def long_log_with_features(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, 0, date, 1.0],
            [0, 3, datetime(2019, 1, 5), 3.0],
            [0, 1, date, 2.0],
            [0, 4, date, 4.0],
            [1, 0, datetime(2020, 1, 5), 4.0],
            [1, 2, datetime(2018, 1, 1), 2.0],
            [1, 6, datetime(2019, 1, 1), 4.0],
            [1, 7, datetime(2020, 1, 1), 4.0],
            [2, 8, date, 3.0],
            [2, 1, date, 2.0],
            [2, 5, datetime(2020, 3, 1), 1.0],
            [2, 6, date, 5.0],
        ],
        schema=["user_idx", "item_idx", "timestamp", "relevance"],
    )


@pytest.fixture
def short_log_with_features(spark):
    date = datetime(2021, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, 2, date, 1.0],
            [0, 4, datetime(2019, 1, 5), 3.0],
            [1, 1, date, 1.0],
            [1, 6, datetime(2018, 1, 1), 2.0],
            [2, 5, date, 3.0],
            [2, 0, date, 2.0],
            [3, 4, date, 5.0],
        ],
        schema=["user_idx", "item_idx", "timestamp", "relevance"],
    )


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame(
        [
            (0, 20.0, -3.0, "M"),
            (1, 30.0, 4.0, "F"),
            (2, 75.0, -1.0, "M"),
        ]
    ).toDF("user_idx", "age", "mood", "gender")


@pytest.fixture
def all_users_features(spark):
    return spark.createDataFrame(
        [
            (0, 20.0, -3.0, "M"),
            (1, 30.0, 4.0, "F"),
            (2, 75.0, -1.0, "M"),
            (3, 35.0, 42.0, "M"),
        ]
    ).toDF("user_idx", "age", "mood", "gender")


@pytest.fixture
def item_features(spark):
    return spark.createDataFrame(
        [
            (0, 4.0, "cat", "black"),
            (1, 10.0, "dog", "green"),
            (2, 7.0, "mouse", "yellow"),
            (3, -1.0, "cat", "yellow"),
            (4, 11.0, "dog", "white"),
            (5, 0.0, "mouse", "yellow"),
        ]
    ).toDF("item_idx", "iq", "class", "color")


@pytest.fixture
def fake_fit_items(spark):
    return spark.createDataFrame(
        [
            (0,),
            (1,),
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
        ]
    ).toDF("item_idx")


@pytest.fixture
def fake_fit_queries(spark):
    return spark.createDataFrame(
        [
            (0,),
            (1,),
            (2,),
            (0,),
            (2,),
            (3,),
        ]
    ).toDF("user_idx")
