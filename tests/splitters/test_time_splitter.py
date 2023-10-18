from typing import List
from datetime import datetime

import pytest
import pandas as pd
import pyspark.sql.functions as F

from replay.splitters import TimeSplitter
from replay.utils import get_spark_session


def _get_column_list(data, column: str) -> List[List]:
    return [[ids[0] for ids in dataframe.select(column).collect()] for dataframe in data]


def _get_column_list_pandas(data, column: str) -> List[List]:
    return [dataframe[column].tolist() for dataframe in data]


def _check_assert(user_ids, item_ids, user_answer, item_answer):
    for idx, item_id in enumerate(item_ids):
        assert sorted(item_id) == sorted(item_answer[idx])
        assert sorted(user_ids[idx]) == sorted(user_answer[idx])


@pytest.fixture(scope="module")
def spark_dataframe_test():
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
    return get_spark_session().createDataFrame(data, schema=columns).withColumn(
        "timestamp", F.to_date("timestamp", "dd-MM-yyyy")
    )


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


@pytest.mark.parametrize(
    "time_threshold, user_answer, item_answer",
    [
        (
            [datetime.strptime("06-01-2020", "%d-%m-%Y")],
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], [2, 2, 2, 2, 2]],
            [[1, 2, 3, 4, 5, 1, 5, 3, 1, 2], [1, 2, 3, 9, 10]],
        ),
        (
            [datetime.strptime("04-01-2020", "%d-%m-%Y"), datetime.strptime("08-01-2020", "%d-%m-%Y")],
            [[1, 1, 1, 3, 3, 3], [1, 1, 2, 2, 3, 3], [2, 2, 2]],
            [[1, 2, 3, 1, 5, 3], [4, 5, 1, 2, 1, 2], [3, 9, 10]],
        ),
    ],
)
@pytest.mark.parametrize(
    "dataset_type",
    [
        ("spark_dataframe_test"),
        ("pandas_dataframe_test"),
    ],
)
def test_time_splitter_without_drops(time_threshold, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = TimeSplitter(
        time_threshold=time_threshold,
        drop_cold_users=False,
        drop_cold_items=False,
    ).split(dataframe)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


@pytest.mark.spark
@pytest.mark.parametrize(
    "time_threshold, user_answer, item_answer",
    [
        (
            [datetime.strptime("06-01-2020", "%d-%m-%Y")],
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], []],
            [[1, 2, 3, 4, 5, 1, 5, 3, 1, 2], []],
        ),
        (
            [datetime.strptime("04-01-2020", "%d-%m-%Y"), datetime.strptime("08-01-2020", "%d-%m-%Y")],
            [[1, 1, 1, 3, 3, 3], [1, 1, 3, 3], []],
            [[1, 2, 3, 1, 5, 3], [4, 5, 1, 2], []],
        ),
    ],
)
@pytest.mark.parametrize(
    "dataset_type",
    [
        ("spark_dataframe_test"),
        ("pandas_dataframe_test"),
    ],
)
def test_time_splitter_drop_users(time_threshold, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = TimeSplitter(
        time_threshold=time_threshold,
        drop_cold_users=True,
        drop_cold_items=False,
    ).split(dataframe)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


@pytest.mark.parametrize(
    "time_threshold, user_answer, item_answer",
    [
        (
            [datetime.strptime("06-01-2020", "%d-%m-%Y")],
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], [2, 2, 2]],
            [[1, 2, 3, 4, 5, 1, 5, 3, 1, 2], [1, 2, 3]],
        ),
        (
            [datetime.strptime("04-01-2020", "%d-%m-%Y"), datetime.strptime("08-01-2020", "%d-%m-%Y")],
            [[1, 1, 1, 3, 3, 3], [1, 2, 2, 3, 3], [2]],
            [[1, 2, 3, 1, 5, 3], [5, 1, 2, 1, 2], [3]],
        ),
    ],
)
@pytest.mark.parametrize(
    "dataset_type",
    [
        ("spark_dataframe_test"),
        ("pandas_dataframe_test"),
    ],
)
def test_time_splitter_drop_items(time_threshold, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = TimeSplitter(
        time_threshold=time_threshold,
        drop_cold_users=False,
        drop_cold_items=True,
    ).split(dataframe)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


@pytest.mark.parametrize(
    "time_threshold, user_answer, item_answer",
    [
        (
            [datetime.strptime("06-01-2020", "%d-%m-%Y")],
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], []],
            [[1, 2, 3, 4, 5, 1, 5, 3, 1, 2], []],
        ),
        (
            [datetime.strptime("04-01-2020", "%d-%m-%Y"), datetime.strptime("08-01-2020", "%d-%m-%Y")],
            [[1, 1, 1, 3, 3, 3], [1, 3, 3], []],
            [[1, 2, 3, 1, 5, 3], [5, 1, 2], []],
        ),
    ],
)
@pytest.mark.parametrize(
    "dataset_type",
    [
        ("spark_dataframe_test"),
        ("pandas_dataframe_test"),
    ],
)
def test_time_splitter_drop_both(time_threshold, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = TimeSplitter(
        time_threshold=time_threshold,
        drop_cold_users=True,
        drop_cold_items=True,
    ).split(dataframe)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


@pytest.mark.parametrize(
    "time_threshold, user_answer, item_answer, session_id_processing_strategy",
    [
        (
            [datetime.strptime("06-01-2020", "%d-%m-%Y")],
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], [2, 2, 2, 2, 2]],
            [[1, 2, 3, 4, 5, 1, 5, 3, 1, 2], [1, 2, 3, 9, 10]],
            "train",
        ),
        (
            [datetime.strptime("06-01-2020", "%d-%m-%Y")],
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], [2, 2, 2, 2, 2]],
            [[1, 2, 3, 4, 5, 1, 5, 3, 1, 2], [1, 2, 3, 9, 10]],
            "test",
        ),
    ],
)
@pytest.mark.parametrize(
    "dataset_type",
    [
        ("spark_dataframe_test"),
        ("pandas_dataframe_test"),
    ],
)
def test_time_splitter_without_drops_with_sessions(
    time_threshold, user_answer, item_answer, session_id_processing_strategy, dataset_type, request
):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = TimeSplitter(
        time_threshold=time_threshold,
        drop_cold_users=False,
        drop_cold_items=False,
        session_id_col="session_id",
        session_id_processing_strategy=session_id_processing_strategy,
    ).split(dataframe)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


@pytest.mark.usefixtures("spark_dataframe_test")
def test_splitter_with_sessions_error(spark_dataframe_test):
    with pytest.raises(NotImplementedError):
        TimeSplitter(
            time_threshold="04-01-2020",
            drop_cold_users=False,
            drop_cold_items=False,
            session_id_col="session_id",
            session_id_processing_strategy="smth",
        ).split(spark_dataframe_test)


def test_original_dataframe_not_change(pandas_dataframe_test):
    original_dataframe = pandas_dataframe_test.copy(deep=True)

    TimeSplitter([datetime.strptime("06-01-2020", "%d-%m-%Y")]).split(original_dataframe)

    assert original_dataframe.equals(pandas_dataframe_test)
