from typing import List

import pandas as pd
import polars as pl
import pytest

from replay.splitters import RatioSplitter
from replay.utils import PYSPARK_AVAILABLE
from tests.utils import spark

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
    return spark.createDataFrame(data, schema=columns).withColumn(
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


@pytest.fixture(scope="module")
def polars_dataframe_test(pandas_dataframe_test):
    return pl.from_pandas(pandas_dataframe_test)


@pytest.fixture(scope="module")
def dataframe_not_implemented(pandas_dataframe_test):
    return pandas_dataframe_test.to_numpy()


@pytest.mark.parametrize(
    "ratio, user_answer, item_answer, split_by_fractions",
    [
        (
            0.5,
            [[1, 1, 2, 2, 3, 3], [1, 1, 1, 2, 2, 2, 3, 3, 3]],
            [[1, 2, 1, 2, 1, 5], [3, 4, 5, 3, 9, 10, 3, 1, 2]],
            True,
        ),
        (
            0.1,
            [[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], [1, 2, 3]],
            [[1, 2, 3, 4, 1, 2, 3, 9, 1, 5, 3, 1], [5, 10, 2]],
            True,
        ),
        (
            0.5,
            [[1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 1, 2, 2, 3, 3]],
            [[1, 2, 3, 1, 2, 3, 1, 5, 3], [4, 5, 9, 10, 1, 2]],
            False,
        ),
        (
            0.1,
            [[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], [1, 2, 3]],
            [[1, 2, 3, 4, 1, 2, 3, 9, 1, 5, 3, 1], [5, 10, 2]],
            False,
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
def test_ratio_splitter_without_drops(ratio, user_answer, item_answer, split_by_fractions, request, dataset_type):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = RatioSplitter(
        test_size=ratio,
        divide_column="user_id",
        query_column="user_id",
        drop_cold_users=False,
        drop_cold_items=False,
        split_by_fractions=split_by_fractions,
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
    "ratio, user_answer, item_answer, min_interactions_per_group, split_by_fractions",
    [
        (
            0.5,
            [[1, 1, 2, 2, 3, 3], [1, 1, 1, 2, 2, 2, 3, 3, 3]],
            [[1, 2, 1, 2, 1, 5], [3, 4, 5, 3, 9, 10, 3, 1, 2]],
            5,
            True,
        ),
        (
            0.5,
            [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], []],
            [[1, 2, 3, 4, 5, 1, 2, 3, 9, 10, 1, 5, 3, 1, 2], []],
            6,
            True,
        ),
        (
            0.5,
            [[1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 1, 2, 2, 3, 3]],
            [[1, 2, 3, 1, 2, 3, 1, 5, 3], [4, 5, 9, 10, 1, 2]],
            5,
            False,
        ),
        (
            0.5,
            [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], []],
            [[1, 2, 3, 4, 5, 1, 2, 3, 9, 10, 1, 5, 3, 1, 2], []],
            6,
            False,
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
def test_ratio_splitter_min_user_interactions(
    ratio, user_answer, item_answer, min_interactions_per_group, split_by_fractions, dataset_type, request
):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = RatioSplitter(
        test_size=ratio,
        divide_column="user_id",
        query_column="user_id",
        drop_cold_users=False,
        drop_cold_items=False,
        min_interactions_per_group=min_interactions_per_group,
        split_by_fractions=split_by_fractions,
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
    "ratio, user_answer, item_answer",
    [
        (
            0.5,
            [[1, 1, 2, 2, 3, 3], [1, 1, 1, 2, 2, 2, 3, 3, 3]],
            [[1, 2, 1, 2, 1, 5], [3, 4, 5, 3, 9, 10, 3, 1, 2]],
        ),
        (
            0.8,
            [[1, 2, 3], [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]],
            [[1, 1, 1], [2, 3, 4, 5, 2, 3, 9, 10, 5, 3, 1, 2]],
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
def test_ratio_splitter_drop_users(ratio, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = RatioSplitter(
        test_size=ratio,
        divide_column="user_id",
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
    "ratio, user_answer, item_answer",
    [
        (
            0.5,
            [[1, 1, 2, 2, 3, 3], [1, 3, 3]],
            [[1, 2, 1, 2, 1, 5], [5, 1, 2]],
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
def test_ratio_splitter_drop_items(ratio, user_answer, item_answer, dataset_type, request):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = RatioSplitter(
        test_size=ratio,
        divide_column="user_id",
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


@pytest.mark.core
def test_ratio_splitter_sanity_check():
    with pytest.raises(ValueError):
        RatioSplitter(test_size=1.4, divide_column="user_id", query_column="user_id")


@pytest.mark.spark
def test_datasets_types_mismatch(spark_dataframe_test, pandas_dataframe_test):
    with pytest.raises(TypeError):
        RatioSplitter(0.1, divide_column="user_id", query_column="user_id")._drop_cold_items_and_users(
            spark_dataframe_test, pandas_dataframe_test
        )


@pytest.mark.parametrize(
    "ratio, user_answer, item_answer, split_by_fractions, session_id_processing_strategy",
    [
        (
            0.1,
            [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], []],
            [[1, 2, 3, 4, 5, 1, 2, 3, 9, 10, 1, 5, 3, 1, 2], []],
            True,
            "train",
        ),
        (
            0.1,
            [[2, 2, 2, 3, 3, 3], [1, 1, 1, 1, 1, 2, 2, 3, 3]],
            [[1, 2, 3, 1, 5, 3], [1, 2, 3, 4, 5, 9, 10, 1, 2]],
            True,
            "test",
        ),
        (
            0.5,
            [[1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3], [2, 2, 3, 3]],
            [[1, 2, 3, 4, 5, 1, 2, 3, 1, 5, 3], [9, 10, 1, 2]],
            False,
            "train",
        ),
        (
            0.5,
            [[2, 2, 2, 3, 3, 3], [1, 1, 1, 1, 1, 2, 2, 3, 3]],
            [[1, 2, 3, 1, 5, 3], [1, 2, 3, 4, 5, 9, 10, 1, 2]],
            False,
            "test",
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
def test_ratio_splitter_without_drops_with_sessions(
    ratio, user_answer, item_answer, split_by_fractions, session_id_processing_strategy, dataset_type, request
):
    dataframe = request.getfixturevalue(dataset_type)

    filtered_dataframe = RatioSplitter(
        test_size=ratio,
        divide_column="user_id",
        query_column="user_id",
        drop_cold_users=False,
        drop_cold_items=False,
        split_by_fractions=split_by_fractions,
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

    RatioSplitter(0.5, query_column="user_id", divide_column="user_id",).split(original_dataframe)

    assert original_dataframe.equals(pandas_dataframe_test)


@pytest.mark.core
def test_not_implemented_dataframe(dataframe_not_implemented):
    with pytest.raises(NotImplementedError):
        RatioSplitter(0.5).split(dataframe_not_implemented)
