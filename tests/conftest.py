from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from replay.data import get_schema
from replay.utils import PYSPARK_AVAILABLE
from replay.utils.session_handler import get_spark_session

if PYSPARK_AVAILABLE:
    INTERACTIONS_SCHEMA = get_schema("user_idx", "item_idx", "timestamp", "relevance")


@pytest.fixture(scope="session")
def spark():
    session = get_spark_session()
    session.sparkContext.setLogLevel("ERROR")
    return session


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def log_pandas(log):
    return log.toPandas()


@pytest.fixture(scope="session")
def log_polars(log):
    return pl.from_pandas(log.toPandas())


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def long_log_with_features_pandas(long_log_with_features):
    return long_log_with_features.toPandas()


@pytest.fixture(scope="session")
def long_log_with_features_polars(long_log_with_features):
    return pl.from_pandas(long_log_with_features.toPandas())


@pytest.fixture(scope="session")
def long_log_with_features_and_none_queries(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [None, 0, date, 1.0],
            [0, 3, datetime(2019, 1, 5), 3.0],
            [0, 1, date, 2.0],
            [0, 4, date, 4.0],
            [1, 0, datetime(2020, 1, 5), 4.0],
            [None, 2, datetime(2018, 1, 1), 2.0],
            [1, 6, datetime(2019, 1, 1), 4.0],
            [1, 7, datetime(2020, 1, 1), 4.0],
            [2, 8, date, 3.0],
            [None, 1, date, 2.0],
            [2, 5, datetime(2020, 3, 1), 1.0],
            [2, 6, date, 5.0],
        ],
        schema=["user_idx", "item_idx", "timestamp", "relevance"],
    )


@pytest.fixture(scope="session")
def long_log_with_features_and_none_queries_pandas(long_log_with_features_and_none_queries):
    return long_log_with_features_and_none_queries.toPandas()


@pytest.fixture(scope="session")
def long_log_with_features_and_none_queries_polars(long_log_with_features_and_none_queries):
    res = pl.from_pandas(long_log_with_features_and_none_queries.toPandas())
    res = res.with_columns(pl.col("user_idx").cast(pl.Int64).alias("user_idx"))
    return res


@pytest.fixture(scope="session")
def long_log_with_features_and_none_items(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, None, date, 1.0],
            [0, 2, datetime(2019, 1, 5), 3.0],
            [0, 1, date, 2.0],
            [0, 4, date, 4.0],
            [1, 0, datetime(2020, 1, 5), 4.0],
            [1, None, datetime(2018, 1, 1), 2.0],
            [1, 6, datetime(2019, 1, 1), 4.0],
            [1, 7, datetime(2020, 1, 1), 4.0],
            [2, 8, date, 3.0],
            [2, 1, date, 2.0],
            [2, None, datetime(2020, 3, 1), 1.0],
            [2, 6, date, 5.0],
        ],
        schema=["user_idx", "item_idx", "timestamp", "relevance"],
    )


@pytest.fixture(scope="session")
def long_log_with_features_and_none_items_pandas(long_log_with_features_and_none_items):
    return long_log_with_features_and_none_items.toPandas()


@pytest.fixture(scope="session")
def long_log_with_features_and_none_items_polars(long_log_with_features_and_none_items):
    return pl.from_pandas(long_log_with_features_and_none_items.toPandas())


@pytest.fixture(scope="session")
def long_log_with_features_and_random_sorted(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [2, 8, date, 3.0],
            [0, 0, date, 1.0],
            [0, 4, date, 4.0],
            [1, 6, datetime(2019, 1, 1), 4.0],
            [0, 3, datetime(2019, 1, 5), 3.0],
            [1, 0, datetime(2020, 1, 5), 4.0],
            [2, 5, datetime(2020, 3, 1), 1.0],
            [1, 2, datetime(2018, 1, 1), 2.0],
            [0, 1, date, 2.0],
            [1, 7, datetime(2020, 1, 1), 4.0],
            [2, 1, date, 2.0],
            [2, 6, date, 5.0],
        ],
        schema=["user_idx", "item_idx", "timestamp", "relevance"],
    )


@pytest.fixture(scope="session")
def long_log_with_features_and_random_sorted_pandas(long_log_with_features_and_random_sorted):
    return long_log_with_features_and_random_sorted.toPandas()


@pytest.fixture(scope="session")
def long_log_with_features_and_random_sorted_polars(long_log_with_features_and_random_sorted):
    return pl.from_pandas(long_log_with_features_and_random_sorted.toPandas())


@pytest.fixture(scope="session")
def long_log_with_features_and_one_query(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [1, 0, date, 1.0],
            [1, 3, datetime(2019, 1, 5), 3.0],
            [1, 1, date, 2.0],
            [1, 4, date, 4.0],
            [1, 0, datetime(2020, 1, 5), 4.0],
            [1, 2, datetime(2018, 1, 1), 2.0],
            [1, 6, datetime(2019, 1, 1), 4.0],
            [1, 7, datetime(2020, 1, 1), 4.0],
            [1, 8, date, 3.0],
            [1, 1, date, 2.0],
            [1, 5, datetime(2020, 3, 1), 1.0],
            [1, 6, date, 5.0],
        ],
        schema=["user_idx", "item_idx", "timestamp", "relevance"],
    )


@pytest.fixture(scope="session")
def long_log_with_features_and_one_query_pandas(long_log_with_features_and_one_query):
    return long_log_with_features_and_one_query.toPandas()


@pytest.fixture(scope="session")
def long_log_with_features_and_one_query_polars(long_log_with_features_and_one_query):
    return pl.from_pandas(long_log_with_features_and_one_query.toPandas())


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def user_features(spark):
    return spark.createDataFrame(
        [
            (0, 20.0, -3.0, "M"),
            (1, 30.0, 4.0, "F"),
            (2, 75.0, -1.0, "M"),
        ]
    ).toDF("user_idx", "age", "mood", "gender")


@pytest.fixture(scope="session")
def user_features_pandas(user_features):
    return user_features.toPandas()


@pytest.fixture(scope="session")
def user_features_polars(user_features):
    return pl.from_pandas(user_features.toPandas())


@pytest.fixture(scope="session")
def all_users_features(spark):
    return spark.createDataFrame(
        [
            (0, 20.0, -3.0, "M"),
            (1, 30.0, 4.0, "F"),
            (2, 75.0, -1.0, "M"),
            (3, 35.0, 42.0, "M"),
        ]
    ).toDF("user_idx", "age", "mood", "gender")


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def item_features_pandas(item_features):
    return item_features.toPandas()


@pytest.fixture(scope="session")
def item_features_polars(item_features):
    return pl.from_pandas(item_features.toPandas())


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def fake_fit_queries_pandas():
    return pd.DataFrame({"user_idx": [0, 1, 2, 0, 2, 3]})


@pytest.fixture(scope="session")
def fake_fit_queries_polars():
    return pl.DataFrame({"user_idx": [0, 1, 2, 0, 2, 3]})
