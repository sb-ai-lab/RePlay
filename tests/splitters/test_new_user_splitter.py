from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from replay.splitters import NewUsersSplitter
from replay.utils import PandasDataFrame

log_data = [
    [1, 3, datetime(2019, 9, 14), 3.0, 1],
    [1, 0, datetime(2019, 9, 14), 3.0, 1],
    [1, 1, datetime(2019, 9, 15), 4.0, 1],
    [0, 3, datetime(2019, 9, 12), 1.0, 1],
    [3, 0, datetime(2019, 9, 12), 1.0, 1],
    [3, 1, datetime(2019, 9, 13), 2.0, 1],
    [2, 0, datetime(2019, 9, 16), 5.0, 1],
    [2, 3, datetime(2019, 9, 16), 5.0, 1],
    [0, 2, datetime(2019, 9, 17), 1.0, 1],
]


@pytest.fixture(scope="module")
def log(spark):
    return spark.createDataFrame(
        log_data,
        schema=["user_id", "item_id", "timestamp", "relevance", "session_id"],
    )


@pytest.fixture(scope="module")
def log_pandas():
    return PandasDataFrame(log_data, columns=["user_id", "item_id", "timestamp", "relevance", "session_id"])


@pytest.fixture(scope="module")
def log_polars(log_pandas):
    return pl.from_pandas(log_pandas)


@pytest.fixture(scope="module")
def log_not_implemented(log_pandas):
    return log_pandas.to_numpy()


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log", marks=pytest.mark.spark),
        pytest.param("log_pandas", marks=pytest.mark.core),
        pytest.param("log_polars", marks=pytest.mark.core),
    ],
)
def test_users_are_cold(dataset_type, request):
    log = request.getfixturevalue(dataset_type)
    splitter = NewUsersSplitter(
        test_size=0.25, query_column="user_id", drop_cold_items=False, session_id_column="session_id"
    )
    train, test = splitter.split(log)

    if isinstance(log, pd.DataFrame):
        train_users = train.user_id
        test_users = test.user_id
    elif isinstance(log, pl.DataFrame):
        train_users = train["user_id"]
        test_users = test["user_id"]
    else:
        train_users = train.toPandas().user_id
        test_users = test.toPandas().user_id

    assert not np.isin(test_users, train_users).any()


@pytest.mark.core
def test_bad_test_size():
    with pytest.raises(ValueError):
        NewUsersSplitter(1.2)


@pytest.mark.core
def test_not_implemented_dataframe(log_not_implemented):
    with pytest.raises(NotImplementedError):
        NewUsersSplitter(0.2).split(log_not_implemented)
