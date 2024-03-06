# pylint: disable-all
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from replay.splitters import TwoStageSplitter
from replay.utils import PandasDataFrame, SparkDataFrame
from tests.utils import spark

log_data = [
    [0, 3, datetime(2019, 9, 12), 1.0, 1],
    [1, 4, datetime(2019, 9, 13), 2.0, 1],
    [2, 6, datetime(2019, 9, 17), 1.0, 1],
    [3, 5, datetime(2019, 9, 17), 1.0, 1],
    [4, 5, datetime(2019, 9, 17), 1.0, 1],
    [0, 5, datetime(2019, 9, 12), 1.0, 1],
    [1, 6, datetime(2019, 9, 13), 2.0, 1],
    [2, 7, datetime(2019, 9, 17), 1.0, 1],
    [3, 8, datetime(2019, 9, 17), 1.0, 1],
    [4, 0, datetime(2019, 9, 17), 1.0, 1],
]


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        log_data,
        schema=["user_id", "item_id", "timestamp", "relevance", "session_id"],
    )


@pytest.fixture
def log_pandas():
    return PandasDataFrame(log_data, columns=["user_id", "item_id", "timestamp", "relevance", "session_id"])


@pytest.fixture
def log_polars(log_pandas):
    return pl.from_pandas(log_pandas)


@pytest.fixture
def log_not_implemented(log_pandas):
    return log_pandas.to_numpy()


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log", marks=pytest.mark.spark),
        pytest.param("log_pandas", marks=pytest.mark.core),
        pytest.param("log_polars", marks=pytest.mark.core),
    ]
)
@pytest.mark.parametrize("fraction", [3, 0.6])
def test_get_test_values(dataset_type, request, fraction):
    log = request.getfixturevalue(dataset_type)
    splitter = TwoStageSplitter(
        first_divide_size=fraction,
        second_divide_size=1,
        first_divide_column="user_id",
        query_column="user_id",
        drop_cold_items=False,
        drop_cold_users=False,
        seed=1234,
    )
    test_users = splitter._get_test_values(log)
    if isinstance(log, pd.DataFrame):
        assert test_users.shape[0] == 3
        assert np.isin([0, 1, 4], test_users["user_id"]).all()
    elif isinstance(log, pl.DataFrame):
        assert test_users.shape[0] == 3
    else:
        assert test_users.count() == 3
        assert np.isin([0, 2, 3], test_users.toPandas().user_id).all()


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log", marks=pytest.mark.spark),
        pytest.param("log_pandas", marks=pytest.mark.core),
        pytest.param("log_polars", marks=pytest.mark.core),
    ]
)
@pytest.mark.parametrize("fraction", [5, 1.0])
def test_user_test_size_exception(dataset_type, request, fraction):
    log = request.getfixturevalue(dataset_type)
    splitter = TwoStageSplitter(
        first_divide_size=fraction,
        second_divide_size=1,
        first_divide_column="user_id",
        query_column="user_id",
        drop_cold_items=False,
        drop_cold_users=False,
    )
    with pytest.raises(ValueError):
        splitter._get_test_values(log)


big_log_data = [
    [0, 3, datetime(2019, 9, 12), 1.0, 1],
    [0, 4, datetime(2019, 9, 13), 2.0, 1],
    [0, 6, datetime(2019, 9, 17), 1.0, 1],
    [0, 5, datetime(2019, 9, 17), 1.0, 1],
    [1, 3, datetime(2019, 9, 12), 1.0, 1],
    [1, 4, datetime(2019, 9, 13), 2.0, 1],
    [1, 5, datetime(2019, 9, 14), 3.0, 1],
    [1, 1, datetime(2019, 9, 15), 4.0, 1],
    [1, 2, datetime(2019, 9, 15), 4.0, 1],
    [2, 3, datetime(2019, 9, 12), 1.0, 1],
    [2, 4, datetime(2019, 9, 13), 2.0, 1],
    [2, 5, datetime(2019, 9, 14), 3.0, 1],
    [2, 1, datetime(2019, 9, 14), 3.0, 1],
    [2, 6, datetime(2019, 9, 17), 1.0, 1],
    [3, 1, datetime(2019, 9, 15), 4.0, 1],
    [3, 0, datetime(2019, 9, 16), 4.0, 1],
    [3, 3, datetime(2019, 9, 17), 4.0, 1],
    [3, 4, datetime(2019, 9, 18), 4.0, 1],
    [3, 7, datetime(2019, 9, 19), 4.0, 1],
    [3, 3, datetime(2019, 9, 20), 4.0, 1],
    [3, 0, datetime(2019, 9, 21), 4.0, 1],
]


@pytest.fixture
def big_log(spark):
    return spark.createDataFrame(
        big_log_data,
        schema=["user_id", "item_id", "timestamp", "relevance", "session_id"],
    )


@pytest.fixture
def big_log_pandas():
    return PandasDataFrame(big_log_data, columns=["user_id", "item_id", "timestamp", "relevance", "session_id"])


@pytest.fixture
def big_log_polars(big_log_pandas):
    return pl.from_pandas(big_log_pandas)


test_sizes = np.arange(0.1, 1, 0.25).tolist() + list(range(1, 5))


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("big_log", marks=pytest.mark.spark),
        pytest.param("big_log_pandas", marks=pytest.mark.core),
        pytest.param("big_log_polars", marks=pytest.mark.core),
    ]
)
@pytest.mark.parametrize("item_test_size", test_sizes)
@pytest.mark.parametrize("shuffle", [True, False])
def test_random_split(dataset_type, request, item_test_size, shuffle):
    big_log = request.getfixturevalue(dataset_type)
    splitter = TwoStageSplitter(
        first_divide_size=0.5,
        second_divide_size=item_test_size,
        first_divide_column="user_id",
        query_column="user_id",
        drop_cold_items=False,
        drop_cold_users=False,
        seed=1234,
        shuffle=shuffle,
    )
    train, test = splitter.split(big_log)

    if isinstance(big_log, pd.DataFrame):
        assert train.shape[0] + test.shape[0] == big_log.shape[0]
        assert len(train.merge(test, on=["user_id", "item_id", "timestamp", "session_id"], how="inner")) == 0

        if isinstance(item_test_size, int):
            #  it's a rough check. for it to be true, item_test_size must be bigger than log length for every user
            num_users = big_log["user_id"].nunique() * 0.5     # only half of users go to test
            assert num_users * item_test_size == test.shape[0]
            assert big_log.shape[0] - num_users * item_test_size == train.shape[0]
    elif isinstance(big_log, pl.DataFrame):
        assert train.shape[0] + test.shape[0] == big_log.shape[0]
        assert len(train.join(test, on=["user_id", "item_id", "timestamp", "session_id"], how="inner")) == 0

        if isinstance(item_test_size, int):
            #  it's a rough check. for it to be true, item_test_size must be bigger than log length for every user
            num_users = len(big_log["user_id"].unique()) * 0.5     # only half of users go to test
            assert num_users * item_test_size == test.shape[0]
            assert big_log.shape[0] - num_users * item_test_size == train.shape[0]
    else:
        assert train.count() + test.count() == big_log.count()
        assert test.intersect(train).count() == 0

        if isinstance(item_test_size, int):
            #  it's a rough check. for it to be true, item_test_size must be bigger than log length for every user
            num_users = big_log.select("user_id").distinct().count() * 0.5
            assert num_users * item_test_size == test.count()
            assert big_log.count() - num_users * item_test_size == train.count()


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("big_log", marks=pytest.mark.spark),
        pytest.param("big_log_pandas", marks=pytest.mark.core),
        pytest.param("big_log_polars", marks=pytest.mark.core),
    ]
)
@pytest.mark.parametrize("item_test_size", [2.0, -1, -50, 2.1, -0.01])
def test_item_test_size_exception(dataset_type, request, item_test_size):
    big_log = request.getfixturevalue(dataset_type)
    splitter = TwoStageSplitter(
        first_divide_size=2,
        second_divide_size=item_test_size,
        first_divide_column="user_id",
        query_column="user_id",
        drop_cold_items=False,
        drop_cold_users=False,
        seed=1234,
    )
    with pytest.raises(ValueError):
        splitter.split(big_log)


log2_data = [
    [0, 0, datetime(2019, 1, 1), 1.0, 1],
    [0, 1, datetime(2019, 1, 2), 1.0, 1],
    [0, 2, datetime(2019, 1, 3), 1.0, 1],
    [0, 3, datetime(2019, 1, 4), 1.0, 1],
    [1, 4, datetime(2020, 2, 5), 1.0, 1],
    [1, 3, datetime(2020, 2, 4), 1.0, 1],
    [1, 2, datetime(2020, 2, 3), 1.0, 1],
    [1, 1, datetime(2020, 2, 2), 1.0, 1],
    [1, 0, datetime(2020, 2, 1), 1.0, 1],
    [2, 0, datetime(1995, 1, 1), 1.0, 1],
    [2, 1, datetime(1995, 1, 2), 1.0, 1],
    [2, 2, datetime(1995, 1, 3), 1.0, 1],
]


@pytest.fixture
def log2(spark):
    return spark.createDataFrame(
        log2_data,
        schema=["user_id", "item_id", "timestamp", "relevance", "session_id"],
    )


@pytest.fixture
def log2_pandas():
    return PandasDataFrame(log2_data, columns=["user_id", "item_id", "timestamp", "relevance", "session_id"])


@pytest.fixture
def log2_polars(log2_pandas):
    return pl.from_pandas(log2_pandas)


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log2", marks=pytest.mark.spark),
        pytest.param("log2_pandas", marks=pytest.mark.core),
        pytest.param("log2_polars", marks=pytest.mark.core),
    ]
)
def test_split_quantity(dataset_type, request):
    log2 = request.getfixturevalue(dataset_type)
    splitter = TwoStageSplitter(
        first_divide_size=0.5,
        second_divide_size=2,
        first_divide_column="user_id",
        query_column="user_id",
        drop_cold_items=False,
        drop_cold_users=False,
    )
    _, test = splitter.split(log2)
    if isinstance(log2, pd.DataFrame):
        num_items = test.user_id.value_counts()
    elif isinstance(log2, pl.DataFrame):
        num_items = test.to_pandas().user_id.value_counts()
    else:
        num_items = test.toPandas().user_id.value_counts()

    assert num_items.nunique() == 1
    assert num_items.unique()[0] == 2


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log2", marks=pytest.mark.spark),
        pytest.param("log2_pandas", marks=pytest.mark.core),
        pytest.param("log2_polars", marks=pytest.mark.core),
    ]
)
def test_split_proportion(dataset_type, request):
    log2 = request.getfixturevalue(dataset_type)
    splitter = TwoStageSplitter(
        first_divide_size=1,
        second_divide_size=0.4,
        first_divide_column="user_id",
        query_column="user_id",
        drop_cold_items=False,
        drop_cold_users=False,
        seed=13,
    )
    _, test = splitter.split(log2)
    if isinstance(log2, pd.DataFrame):
        num_items = test.user_id.value_counts()
        assert num_items[1] == 2
    elif isinstance(log2, pl.DataFrame):
        num_items = test.select("user_id").unique()
        assert num_items.height == 1
    else:
        num_items = test.toPandas().user_id.value_counts()
        assert num_items[0] == 1


@pytest.mark.core
def test_not_implemented_dataframe(log_not_implemented):
    with pytest.raises(NotImplementedError):
        TwoStageSplitter(
            first_divide_size=1,
            second_divide_size=0.4,
        ).split(log_not_implemented)

    with pytest.raises(NotImplementedError):
        TwoStageSplitter(
            first_divide_size=1,
            second_divide_size=2,
        ).split(log_not_implemented)
