from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import polars as pl
import pytest

from replay.splitters import TimeSplitter
from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as F


def _get_column_list(data, column: str) -> List[List]:
    return [[ids[0] for ids in dataframe.select(column).collect()] for dataframe in data]


def _get_column_list_pandas(data, column: str) -> List[List]:
    return [dataframe[column].tolist() for dataframe in data]


def _get_column_list_polars(data, column: str) -> List[List]:
    return [dataframe[column].to_list() for dataframe in data]


def _check_assert(user_ids, item_ids, user_answer, item_answer):
    for idx, item_id in enumerate(item_ids):
        assert sorted(item_id) == sorted(item_answer[idx])
        assert sorted(user_ids[idx]) == sorted(user_answer[idx])


@pytest.fixture()
@pytest.mark.usefixtures("spark")
def spark_dataframe_test(spark):
    columns = ["user_id", "item_id", "timestamp", "session_id"]
    data = [
        (1, 1, "01-01-2020", 1),
        (1, 2, "02-01-2020", 1),
        (1, 3, "03-01-2020", 1),
        (1, 4, "04-01-2020", 1),
        (1, 5, "05-01-2020", 1),
        (2, 1, "06-01-2020", 2),
        (2, 2, "07-01-2020", 2),
        (2, 3, "08-01-2020", 3),
        (2, 9, "09-01-2020", 4),
        (2, 10, "10-01-2020", 4),
        (3, 1, "01-01-2020", 5),
        (3, 5, "02-01-2020", 5),
        (3, 3, "03-01-2020", 5),
        (3, 1, "04-01-2020", 6),
        (3, 2, "05-01-2020", 6),
    ]
    return spark.createDataFrame(data, schema=columns).withColumn("timestamp", F.to_date("timestamp", "dd-MM-yyyy"))


@pytest.fixture(scope="module")
def pandas_dataframe_test():
    columns = ["user_id", "item_id", "timestamp", "session_id"]
    data = [
        (1, 1, "01-01-2020", 1),
        (1, 2, "02-01-2020", 1),
        (1, 3, "03-01-2020", 1),
        (1, 4, "04-01-2020", 1),
        (1, 5, "05-01-2020", 1),
        (2, 1, "06-01-2020", 2),
        (2, 2, "07-01-2020", 2),
        (2, 3, "08-01-2020", 3),
        (2, 9, "09-01-2020", 4),
        (2, 10, "10-01-2020", 4),
        (3, 1, "01-01-2020", 5),
        (3, 5, "02-01-2020", 5),
        (3, 3, "03-01-2020", 5),
        (3, 1, "04-01-2020", 6),
        (3, 2, "05-01-2020", 6),
    ]

    dataframe = pd.DataFrame(data, columns=columns)
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], format="%d-%m-%Y")

    return dataframe


@pytest.fixture(scope="module")
def polars_dataframe_test(pandas_dataframe_test):
    return pl.from_pandas(pandas_dataframe_test)


log_data = [
    [0, 0, datetime(2019, 9, 12), 1.0],
    [0, 1, datetime(2019, 9, 13), 2.0],
    [1, 0, datetime(2019, 9, 14), 3.0],
    [1, 1, datetime(2019, 9, 15), 4.0],
    [2, 0, datetime(2019, 9, 16), 5.0],
    [0, 2, datetime(2019, 9, 17), 1.0],
]


@pytest.fixture()
@pytest.mark.usefixtures("spark")
def log(spark):
    return spark.createDataFrame(
        log_data,
        schema=["user_id", "item_id", "timestamp", "relevance"],
    )


@pytest.fixture()
def log_pandas():
    return PandasDataFrame(log_data, columns=["user_id", "item_id", "timestamp", "relevance"])


@pytest.fixture()
def log_polars(log_pandas):
    return pl.from_pandas(log_pandas)


@pytest.fixture()
def log_not_implemented(log_pandas):
    return log_pandas.to_numpy()


@pytest.mark.parametrize(
    "time_threshold, user_answer, item_answer",
    [
        (
            datetime.strptime("06-01-2020", "%d-%m-%Y"),
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], [2, 2, 2, 2, 2]],
            [[1, 2, 3, 4, 5, 1, 5, 3, 1, 2], [1, 2, 3, 9, 10]],
        ),
    ],
)
@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark_dataframe_test", marks=pytest.mark.spark),
        pytest.param("pandas_dataframe_test", marks=pytest.mark.core),
        pytest.param("polars_dataframe_test", marks=pytest.mark.core),
    ],
)
def test_time_splitter_without_drops(time_threshold, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = TimeSplitter(
        time_threshold=time_threshold,
        query_column="user_id",
        drop_cold_users=False,
        drop_cold_items=False,
    ).split(dataframe)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    elif dataset_type == "polars_dataframe_test":
        item_ids = _get_column_list_polars(filtered_dataframe, "item_id")
        user_ids = _get_column_list_polars(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


@pytest.mark.parametrize(
    "time_threshold, user_answer, item_answer",
    [
        (
            datetime.strptime("06-01-2020", "%d-%m-%Y"),
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], []],
            [[1, 2, 3, 4, 5, 1, 5, 3, 1, 2], []],
        ),
    ],
)
@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark_dataframe_test", marks=pytest.mark.spark),
        pytest.param("pandas_dataframe_test", marks=pytest.mark.core),
        pytest.param("polars_dataframe_test", marks=pytest.mark.core),
    ],
)
def test_time_splitter_drop_users(time_threshold, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = TimeSplitter(
        time_threshold=time_threshold,
        query_column="user_id",
        drop_cold_users=True,
        drop_cold_items=False,
    ).split(dataframe)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    elif dataset_type == "polars_dataframe_test":
        item_ids = _get_column_list_polars(filtered_dataframe, "item_id")
        user_ids = _get_column_list_polars(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


@pytest.mark.parametrize(
    "time_threshold, user_answer, item_answer",
    [
        (
            datetime.strptime("06-01-2020", "%d-%m-%Y"),
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], [2, 2, 2]],
            [[1, 2, 3, 4, 5, 1, 5, 3, 1, 2], [1, 2, 3]],
        ),
    ],
)
@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark_dataframe_test", marks=pytest.mark.spark),
        pytest.param("pandas_dataframe_test", marks=pytest.mark.core),
        pytest.param("polars_dataframe_test", marks=pytest.mark.core),
    ],
)
def test_time_splitter_drop_items(time_threshold, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = TimeSplitter(
        time_threshold=time_threshold,
        query_column="user_id",
        drop_cold_users=False,
        drop_cold_items=True,
    ).split(dataframe)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    elif dataset_type == "polars_dataframe_test":
        item_ids = _get_column_list_polars(filtered_dataframe, "item_id")
        user_ids = _get_column_list_polars(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


@pytest.mark.parametrize(
    "time_threshold, user_answer, item_answer",
    [
        (
            datetime.strptime("06-01-2020", "%d-%m-%Y"),
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], []],
            [[1, 2, 3, 4, 5, 1, 5, 3, 1, 2], []],
        ),
    ],
)
@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark_dataframe_test", marks=pytest.mark.spark),
        pytest.param("pandas_dataframe_test", marks=pytest.mark.core),
        pytest.param("polars_dataframe_test", marks=pytest.mark.core),
    ],
)
def test_time_splitter_drop_both(time_threshold, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = TimeSplitter(
        time_threshold=time_threshold,
        query_column="user_id",
        drop_cold_users=True,
        drop_cold_items=True,
    ).split(dataframe)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    elif dataset_type == "polars_dataframe_test":
        item_ids = _get_column_list_polars(filtered_dataframe, "item_id")
        user_ids = _get_column_list_polars(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


@pytest.mark.parametrize(
    "time_threshold, user_answer, item_answer, session_id_processing_strategy",
    [
        (
            datetime.strptime("06-01-2020", "%d-%m-%Y"),
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], [2, 2, 2, 2, 2]],
            [[1, 2, 3, 4, 5, 1, 5, 3, 1, 2], [1, 2, 3, 9, 10]],
            "train",
        ),
    ],
)
@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark_dataframe_test", marks=pytest.mark.spark),
        pytest.param("pandas_dataframe_test", marks=pytest.mark.core),
        pytest.param("polars_dataframe_test", marks=pytest.mark.core),
    ],
)
def test_time_splitter_without_drops_with_sessions(
    time_threshold, user_answer, item_answer, session_id_processing_strategy, dataset_type, request
):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = TimeSplitter(
        time_threshold=time_threshold,
        query_column="user_id",
        drop_cold_users=False,
        drop_cold_items=False,
        session_id_column="session_id",
        session_id_processing_strategy=session_id_processing_strategy,
    ).split(dataframe)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    elif dataset_type == "polars_dataframe_test":
        item_ids = _get_column_list_polars(filtered_dataframe, "item_id")
        user_ids = _get_column_list_polars(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


def test_original_dataframe_not_change(pandas_dataframe_test):
    original_dataframe = pandas_dataframe_test.copy(deep=True)

    TimeSplitter(datetime.strptime("06-01-2020", "%d-%m-%Y"), query_column="user_id").split(original_dataframe)

    assert original_dataframe.equals(pandas_dataframe_test)


@pytest.fixture()
def split_date():
    return datetime(2019, 9, 15)


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log", marks=pytest.mark.spark),
        pytest.param("log_pandas", marks=pytest.mark.core),
        pytest.param("log_polars", marks=pytest.mark.core),
    ],
)
def test_split(dataset_type, request, split_date):
    log = request.getfixturevalue(dataset_type)
    splitter = TimeSplitter(
        split_date,
        drop_cold_items=False,
        drop_cold_users=False,
        query_column="user_id",
    )
    train, test = splitter.split(log)

    if dataset_type in ["log_pandas", "log_polars"]:
        train_max_date = train["timestamp"].max()
        test_min_date = test["timestamp"].min()
    else:
        train_max_date = train.toPandas().timestamp.max()
        test_min_date = test.toPandas().timestamp.min()

    assert train_max_date < split_date
    assert test_min_date >= split_date


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log", marks=pytest.mark.spark),
        pytest.param("log_pandas", marks=pytest.mark.core),
        pytest.param("log_polars", marks=pytest.mark.core),
    ],
)
def test_string(dataset_type, request, split_date):
    log = request.getfixturevalue(dataset_type)
    splitter = TimeSplitter(
        split_date,
        drop_cold_items=False,
        drop_cold_users=False,
        query_column="user_id",
    )
    train_by_date, test_by_date = splitter.split(log)

    str_date = split_date.strftime("%Y-%m-%d")
    splitter = TimeSplitter(
        str_date,
        drop_cold_items=False,
        drop_cold_users=False,
        time_column_format="%Y-%m-%d",
        query_column="user_id",
    )
    train_by_str, test_by_str = splitter.split(log)

    int_date = int(split_date.timestamp())
    if dataset_type == "log_pandas":
        log["timestamp"] = (log["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    elif dataset_type == "log_polars":
        log = log.with_columns(pl.col("timestamp").dt.epoch("s"))
    else:
        log = log.withColumn("timestamp", log["timestamp"].cast("bigint"))
    splitter = TimeSplitter(int_date, drop_cold_items=False, drop_cold_users=False, query_column="user_id")
    train_by_int, test_by_int = splitter.split(log)

    if dataset_type in ["log_pandas", "log_polars"]:
        assert train_by_date.shape[0] == train_by_str.shape[0]
        assert test_by_date.shape[0] == test_by_str.shape[0]

        assert train_by_date.shape[0] == train_by_int.shape[0]
        assert test_by_date.shape[0] == test_by_int.shape[0]
    else:
        assert train_by_date.count() == train_by_str.count()
        assert test_by_date.count() == test_by_str.count()

        assert train_by_date.count() == train_by_int.count()
        assert test_by_date.count() == test_by_int.count()


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log", marks=pytest.mark.spark),
        pytest.param("log_pandas", marks=pytest.mark.core),
        pytest.param("log_polars", marks=pytest.mark.core),
    ],
)
def test_proportion(dataset_type, request):
    log = request.getfixturevalue(dataset_type)
    test_size = 0.15
    splitter = TimeSplitter(
        test_size,
        query_column="user_id",
    )
    train, test = splitter.split(log)

    if dataset_type in ["log_pandas", "log_polars"]:
        train_max_date = train["timestamp"].max()
        test_min_date = test["timestamp"].min()
    else:
        train_max_date = train.toPandas().timestamp.max()
        test_min_date = test.toPandas().timestamp.min()

    split_date = datetime(2019, 9, 17)

    assert train_max_date < split_date
    assert test_min_date >= split_date
    if dataset_type in ["log_pandas", "log_polars"]:
        proportion = test.shape[0] / log.shape[0]
    else:
        proportion = test.count() / log.count()

    assert np.isclose(proportion, test_size, atol=0.1)


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log", marks=pytest.mark.spark),
        pytest.param("log_pandas", marks=pytest.mark.core),
        pytest.param("log_polars", marks=pytest.mark.core),
    ],
)
def test_drop_cold_items(dataset_type, request, split_date):
    log = request.getfixturevalue(dataset_type)
    splitter = TimeSplitter(split_date, drop_cold_items=True, drop_cold_users=False, query_column="user_id")
    train, test = splitter.split(log)

    if dataset_type in ["log_pandas", "log_polars"]:
        train_items = train["item_id"]
        test_items = test["item_id"]
    else:
        train_items = train.toPandas().item_id
        test_items = test.toPandas().item_id

    assert np.isin(test_items, train_items).all()


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log", marks=pytest.mark.spark),
        pytest.param("log_pandas", marks=pytest.mark.core),
        pytest.param("log_polars", marks=pytest.mark.core),
    ],
)
def test_drop_cold_users(dataset_type, request, split_date):
    log = request.getfixturevalue(dataset_type)
    splitter = TimeSplitter(
        split_date,
        drop_cold_items=False,
        drop_cold_users=True,
        query_column="user_id",
    )
    train, test = splitter.split(log)

    if dataset_type in ["log_pandas", "log_polars"]:
        train_users = train["user_id"]
        test_users = test["user_id"]
    else:
        train_users = train.toPandas().user_id
        test_users = test.toPandas().user_id

    assert np.isin(test_users, train_users).all()


def test_proportion_splitting_out_of_range():
    with pytest.raises(ValueError):
        TimeSplitter(
            1.2,
            query_column="user_id",
        )


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log", marks=pytest.mark.spark),
        pytest.param("log_pandas", marks=pytest.mark.core),
        pytest.param("log_polars", marks=pytest.mark.core),
    ],
)
def test_wrong_threshold_format_passed(dataset_type, request, split_date):
    log = request.getfixturevalue(dataset_type)
    str_date = split_date.strftime("%Y-%m-%d")
    splitter = TimeSplitter(str_date, drop_cold_items=False, drop_cold_users=False, query_column="user_id")
    with pytest.raises(ValueError):
        splitter.split(log)


@pytest.mark.core
def test_not_implemented_dataframe(log_not_implemented):
    with pytest.raises(NotImplementedError):
        TimeSplitter(0.5).split(log_not_implemented)
