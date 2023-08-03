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

from replay.data import REC_SCHEMA, LOG_SCHEMA
from replay.utils.session_handler import get_spark_session


def assertDictAlmostEqual(d1: Dict, d2: Dict) -> None:
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        assert_allclose(d1[key], d2[key])


@pytest.fixture
def spark():
    return get_spark_session(1, 1)


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
        schema=LOG_SCHEMA,
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
        schema=LOG_SCHEMA,
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
        schema=LOG_SCHEMA,
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
        [(0, 20.0, -3.0, "M"), (1, 30.0, 4.0, "F"), (2, 75.0, -1.0, "M"),]
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
