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
import torch

from replay.data import REC_SCHEMA, LOG_SCHEMA
from replay.utils.session_handler import get_spark_session

from replay.models.ddpg import (
    ActorDRR,
    CriticDRR,
    StateReprModule,
)


def assertDictAlmostEqual(d1: Dict, d2: Dict) -> None:
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        assert_allclose(d1[key], d2[key])


@pytest.fixture
def spark():
    session = get_spark_session(1, 1)
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
        [
            (0, 20.0, -3.0, "M"),
            (1, 30.0, 4.0, "F"),
            (2, 75.0, -1.0, "M"),
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


DDPG_PARAMS = [
    dict(
        state_repr_dim=1,
        action_emb_dim=1,
        hidden_dim=1,
        heads_num=1,
        heads_q=0.5,
        user_num=1,
        item_num=1,
        embedding_dim=1,
        memory_size=1,
        device=torch.device("cpu"),
        env_gamma_alpha=1,
        min_trajectory_len=10,
    ),
    dict(
        state_repr_dim=10,
        action_emb_dim=10,
        hidden_dim=1,
        heads_num=15,
        heads_q=0.1,
        user_num=10,
        item_num=10,
        embedding_dim=10,
        memory_size=10,
        device=torch.device("cpu"),
        env_gamma_alpha=1,
        min_trajectory_len=10,
    ),
]

HEADER = ["user_idx", "item_idx", "relevance"]


def matrix_to_df(matrix):
    x1 = np.repeat(np.arange(matrix.shape[0]), matrix.shape[1])
    x2 = np.tile(np.arange(matrix.shape[1]), matrix.shape[0])
    x3 = matrix.flatten()

    return pd.DataFrame(np.array([x1, x2, x3]).T, columns=HEADER)


DF_CASES = [
    matrix_to_df(np.zeros((1, 1), dtype=int)),
    matrix_to_df(np.ones((1, 1), dtype=int)),
    matrix_to_df(np.zeros((10, 10), dtype=int)),
    matrix_to_df(np.ones((10, 10), dtype=int)),
    matrix_to_df(np.random.choice([0, 1], size=(10, 10), p=[0.9, 0.1])),
    # pd.DataFrame(
    #     np.array(
    #         [
    #             [1, 2, 1],
    #             [3, 4, 0],
    #             [7, 9, 1],
    #             [11, 10, 0],
    #             [11, 4, 1],
    #             [7, 10, 1],
    #         ]
    #     ),
    #     columns=HEADER,
    # ),
]


@pytest.fixture(params=DDPG_PARAMS)
def ddpg_critic_param(request):
    param = request.param
    return (
        CriticDRR(
            state_repr_dim=param["state_repr_dim"],
            action_emb_dim=param["action_emb_dim"],
            hidden_dim=param["hidden_dim"],
            heads_num=param["heads_num"],
            heads_q=param["heads_q"],
        ),
        param,
    )


@pytest.fixture(params=DDPG_PARAMS)
def ddpg_actor_param(request):
    param = request.param
    return (
        ActorDRR(
            user_num=param["user_num"],
            item_num=param["item_num"],
            embedding_dim=param["embedding_dim"],
            hidden_dim=param["hidden_dim"],
            memory_size=param["memory_size"],
            env_gamma_alpha=param["env_gamma_alpha"],
            device=param["device"],
            min_trajectory_len=param["min_trajectory_len"],
        ),
        param,
    )


@pytest.fixture(params=DDPG_PARAMS)
def ddpg_state_repr_param(request):
    param = request.param
    return (
        StateReprModule(
            user_num=param["user_num"],
            item_num=param["item_num"],
            embedding_dim=param["embedding_dim"],
            memory_size=param["memory_size"],
        ),
        param,
    )


BATCH_SIZES = [1, 2, 3, 10, 15]
