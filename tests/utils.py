# pylint: skip-file
import os
import re

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pyspark.ml.linalg import DenseVector
from pyspark.sql import DataFrame

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


@pytest.fixture
def long_log_with_features(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            ["u1", "i1", date, 1.0],
            ["u1", "i4", datetime(2019, 1, 5), 3.0],
            ["u1", "i2", date, 2.0],
            ["u1", "i5", date, 4.0],
            ["u2", "i1", date, 1.0],
            ["u2", "i3", datetime(2018, 1, 1), 2.0],
            ["u2", "i7", datetime(2019, 1, 1), 4.0],
            ["u2", "i8", datetime(2020, 1, 1), 4.0],
            ["u3", "i9", date, 3.0],
            ["u3", "i2", date, 2.0],
            ["u3", "i6", datetime(2020, 3, 1), 1.0],
            ["u3", "i7", date, 5.0],
        ],
        schema=["user_id", "item_id", "timestamp", "relevance"],
    )


@pytest.fixture
def short_log_with_features(spark):
    date = datetime(2021, 1, 1)
    return spark.createDataFrame(
        data=[
            ["u1", "i3", date, 1.0],
            ["u1", "i7", datetime(2019, 1, 5), 3.0],
            ["u2", "i2", date, 1.0],
            ["u2", "i10", datetime(2018, 1, 1), 2.0],
            ["u3", "i8", date, 3.0],
            ["u3", "i1", date, 2.0],
            ["u4", "i7", date, 5.0],
        ],
        schema=["user_id", "item_id", "timestamp", "relevance"],
    )


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame(
        [("u1", 20.0, -3.0, "M"), ("u2", 30.0, 4.0, "F")]
    ).toDF("user_id", "age", "mood", "gender")


@pytest.fixture
def item_features(spark):
    return spark.createDataFrame(
        [
            ("i1", 4.0, "cat", "black"),
            ("i2", 10.0, "dog", "green"),
            ("i3", 7.0, "mouse", "yellow"),
            ("i4", -1.0, "cat", "yellow"),
            ("i5", 11.0, "dog", "white"),
            ("i6", 0.0, "mouse", "yellow"),
        ]
    ).toDF("item_id", "iq", "class", "color")


def unify_dataframe(data_frame: DataFrame):
    pandas_df = data_frame.toPandas()
    columns_to_sort_by: List[str] = []

    if len(pandas_df) == 0:
        columns_to_sort_by = pandas_df.columns
    else:
        for column in pandas_df.columns:
            if not type(pandas_df[column][0]) in {
                DenseVector,
                list,
                np.ndarray,
            }:
                columns_to_sort_by.append(column)

    return (
        pandas_df[sorted(data_frame.columns)]
        .sort_values(by=sorted(columns_to_sort_by))
        .reset_index(drop=True)
    )


def sparkDataFrameEqual(df1: DataFrame, df2: DataFrame):
    return pd.testing.assert_frame_equal(
        unify_dataframe(df1), unify_dataframe(df2), check_like=True
    )


def sparkDataFrameNotEqual(df1: DataFrame, df2: DataFrame):
    try:
        sparkDataFrameEqual(df1, df2)
    except AssertionError:
        pass
    else:
        raise AssertionError("spark dataframes are equal")


def del_files_by_pattern(directory: str, pattern: str) -> None:
    """
    Deletes files by pattern
    """
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            os.remove(os.path.join(directory, filename))


def find_file_by_pattern(directory: str, pattern: str) -> Optional[str]:
    """
    Returns path to first found file, if exists
    """
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            return os.path.join(directory, filename)
    return None
