from typing import List

import pytest
import numpy as np
import pandas as pd
import pyspark.sql.functions as F

from replay.splitters import LastNSplitter
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


@pytest.mark.parametrize("strategy", ["interacitons", "INTERACTIONS", "interaction", "second"])
def test_lastnsplitter_wrong_strategy(strategy):
    with pytest.raises(ValueError):
        LastNSplitter(N=1, strategy=strategy, divide_column="user_id", query_column="user_id")


@pytest.mark.parametrize(
    "n, user_answer, item_answer",
    [
        (
            5,
            [[], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]],
            [[], [1, 2, 3, 4, 5, 1, 2, 3, 9, 10, 1, 5, 3, 1, 2]],
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
def test_last_n_interactions_splitter_without_drops(n, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = LastNSplitter(
        N=n,
        divide_column="user_id",
        query_column="user_id",
        strategy="interactions",
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


@pytest.mark.parametrize(
    "n, user_answer, item_answer",
    [
        (
            5,
            [[], []],
            [[], []],
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
def test_last_n_interactions_splitter_drop_users(n, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = LastNSplitter(
        N=n,
        divide_column="user_id",
        query_column="user_id",
        strategy="interactions",
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
    "n, user_answer, item_answer",
    [
        (
            3,
            [[1, 1, 2, 2, 3, 3], [1, 3, 3]],
            [[1, 2, 1, 2, 1, 5], [5, 1, 2]],
        ),
        (
            4,
            [[1, 2, 3], [3]],
            [[1, 1, 1], [1]],
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
def test_last_n_interactions_splitter_drop_items(n, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = LastNSplitter(
        N=n,
        divide_column="user_id",
        query_column="user_id",
        strategy="interactions",
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
    "n, user_answer, item_answer",
    [
        (
            4,
            [[1, 2, 3], [3]],
            [[1, 1, 1], [1]],
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
def test_last_n_interactions_splitter_drop_both(n, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = LastNSplitter(
        N=n,
        divide_column="user_id",
        query_column="user_id",
        strategy="interactions",
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
    "seconds, user_answer, item_answer",
    [
        (
            86400,
            [[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], [1, 2, 3]],
            [[1, 2, 3, 4, 1, 2, 3, 9, 1, 5, 3, 1], [5, 10, 2]],
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
@pytest.mark.parametrize("to_unix_timestamp", [False, True])
def test_last_n_seconds_splitter_without_drops(
    seconds, user_answer, item_answer, to_unix_timestamp, dataset_type, request
):
    dataframe = request.getfixturevalue(dataset_type)

    dataframe_to_split = dataframe
    if dataset_type == "spark_dataframe_test" and to_unix_timestamp:
        dataframe_to_split = dataframe_to_split.withColumn(
            "timestamp", F.unix_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss")
        )
    filtered_dataframe = LastNSplitter(
        N=seconds,
        divide_column="user_id",
        query_column="user_id",
        strategy="timedelta",
        time_column_format="dd-MM-yyyy",
        drop_cold_users=False,
        drop_cold_items=False,
    ).split(dataframe_to_split)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


@pytest.mark.parametrize(
    "seconds, user_answer, item_answer",
    [
        (
            86400,
            [[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], [1, 2, 3]],
            [[1, 2, 3, 4, 1, 2, 3, 9, 1, 5, 3, 1], [5, 10, 2]],
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
def test_last_n_seconds_splitter_drop_users(seconds, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = LastNSplitter(
        N=seconds,
        divide_column="user_id",
        query_column="user_id",
        strategy="timedelta",
        time_column_format="dd-MM-yyyy",
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
    "seconds, user_answer, item_answer",
    [
        (
            86400,
            [[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], [1, 3]],
            [[1, 2, 3, 4, 1, 2, 3, 9, 1, 5, 3, 1], [5, 2]],
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
def test_last_n_seconds_splitter_drop_items(seconds, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = LastNSplitter(
        N=seconds,
        divide_column="user_id",
        query_column="user_id",
        strategy="timedelta",
        time_column_format="dd-MM-yyyy",
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
    "seconds, user_answer, item_answer",
    [
        (
            86400 * 3,
            [[1, 1, 2, 2, 3, 3], [1, 3, 3]],
            [[1, 2, 1, 2, 1, 5], [5, 1, 2]],
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
def test_last_n_seconds_splitter_drop_both(seconds, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = LastNSplitter(
        N=seconds,
        divide_column="user_id",
        query_column="user_id",
        strategy="timedelta",
        time_column_format="dd-MM-yyyy",
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
    "n, user_answer, item_answer, session_id_processing_strategy",
    [
        (
            5,
            [[], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]],
            [[], [1, 2, 3, 4, 5, 1, 2, 3, 9, 10, 1, 5, 3, 1, 2]],
            "train",
        ),
        (
            5,
            [[], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]],
            [[], [1, 2, 3, 4, 5, 1, 2, 3, 9, 10, 1, 5, 3, 1, 2]],
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
def test_last_n_interactions_splitter_without_drops_with_sessions(
    n, user_answer, item_answer, session_id_processing_strategy, dataset_type, request
):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = LastNSplitter(
        N=n,
        divide_column="user_id",
        query_column="user_id",
        strategy="interactions",
        drop_cold_users=False,
        drop_cold_items=False,
        session_id_column="session_id",
        session_id_processing_strategy=session_id_processing_strategy,
    ).split(dataframe)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


@pytest.mark.parametrize(
    "seconds, user_answer, item_answer, session_id_processing_strategy",
    [
        (
            86400,
            [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], []],
            [[1, 2, 3, 4, 5, 1, 2, 9, 10, 3, 1, 5, 3, 1, 2], []],
            "train",
        ),
        (
            86400,
            [[2, 2, 2, 3, 3, 3], [1, 1, 1, 1, 1, 2, 2, 3, 3]],
            [[1, 2, 3, 1, 5, 3], [1, 2, 3, 4, 5, 9, 10, 1, 2]],
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
def test_last_n_seconds_splitter_without_drops_with_sessions(
    seconds, user_answer, item_answer, session_id_processing_strategy, dataset_type, request
):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = LastNSplitter(
        N=seconds,
        divide_column="user_id",
        query_column="user_id",
        strategy="timedelta",
        time_column_format="dd-MM-yyyy",
        drop_cold_users=False,
        drop_cold_items=False,
        session_id_column="session_id",
        session_id_processing_strategy=session_id_processing_strategy,
    ).split(dataframe)

    if dataset_type == "pandas_dataframe_test":
        item_ids = _get_column_list_pandas(filtered_dataframe, "item_id")
        user_ids = _get_column_list_pandas(filtered_dataframe, "user_id")
    else:
        item_ids = _get_column_list(filtered_dataframe, "item_id")
        user_ids = _get_column_list(filtered_dataframe, "user_id")

    _check_assert(user_ids, item_ids, user_answer, item_answer)


@pytest.mark.parametrize(
    "dataset_type, result_type",
    [
        ("spark_dataframe_test", "bigint"),
        ("pandas_dataframe_test", np.dtype("int64")),
    ],
)
def test_last_n_seconds_to_unix_timestamp(dataset_type, result_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    dataframe_splitted = LastNSplitter(
        N=86400,
        divide_column="user_id",
        query_column="user_id",
        strategy="timedelta",
        timestamp_column="timestamp",
        time_column_format="dd-MM-yyyy",
        drop_cold_users=False,
        drop_cold_items=False,
        session_id_column="session_id",
    )._to_unix_timestamp(dataframe)

    assert dict(dataframe_splitted.dtypes)["timestamp"] == result_type


def test_invalid_unix_timestamp(pandas_dataframe_test):
    pandas_dataframe_test_formated_time = LastNSplitter(
        N=86400,
        divide_column="user_id",
        query_column="user_id",
        strategy="timedelta",
        timestamp_column="item_id",
        time_column_format="dd-MM-yyyy",
        drop_cold_users=False,
        drop_cold_items=False,
        session_id_column="session_id",
    )._to_unix_timestamp(pandas_dataframe_test)

    assert dict(pandas_dataframe_test_formated_time.dtypes)["timestamp"] == np.dtype("<M8[ns]")


def test_original_dataframe_not_change(pandas_dataframe_test):
    original_dataframe = pandas_dataframe_test.copy(deep=True)

    LastNSplitter(5, divide_column="user_id", query_column="user_id").split(original_dataframe)

    assert original_dataframe.equals(pandas_dataframe_test)
