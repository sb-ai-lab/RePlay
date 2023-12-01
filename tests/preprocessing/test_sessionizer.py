from typing import List

import pytest

from replay.preprocessing import Sessionizer
from replay.utils import SparkDataFrame


def _get_column_list(data, column: str) -> List[List]:
    return [ids[0] for ids in data.select(column).collect()]


def _get_column_list_pandas(data, column: str) -> List[List]:
    return data[column].tolist()


@pytest.mark.parametrize(
    "time_column, session_gap, min_inter_per_session, max_inter_per_session, user_answer, item_answer, sessions_count",
    [
        ("timestamp", 1, 3, 4, [], [], 0),
        ("timestamp", 1, 1, 2, [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3], [1, 2, 3, 1, 2, 3, 9, 1, 5, 3, 1, 2], 12),
        ("timestamp", 86400, 1, 5, [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3], [1, 2, 3, 1, 2, 3, 9, 1, 5, 3, 1, 2], 12),
        ("timestamp", 86400, 3, 3, [], [], 0),
        ("spark_date", 1, 1, 2, [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3], [1, 2, 3, 1, 2, 3, 9, 1, 5, 3, 1, 2], 12),
        ("spark_date", 0.5, 1, 5, [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3], [1, 2, 3, 1, 2, 3, 9, 1, 5, 3, 1, 2], 12),
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("dataframe_sessionizer", marks=pytest.mark.spark),
        pytest.param("dataframe_sessionizer_pandas", marks=pytest.mark.core),
    ],
)
def test_sessionizer_interactions_per_session(
    time_column,
    session_gap,
    min_inter_per_session,
    max_inter_per_session,
    user_answer,
    item_answer,
    sessions_count,
    dataset,
    request,
):
    dataframe = request.getfixturevalue(dataset)
    is_spark = isinstance(dataframe, SparkDataFrame)
    time_column = "pandas_date" if is_spark is False and time_column == "spark_date" else time_column

    dataframe_with_sessions = Sessionizer(
        time_column=time_column,
        session_gap=session_gap,
        time_column_format="dd-MM-yyyy",
        min_inter_per_session=min_inter_per_session,
        max_inter_per_session=max_inter_per_session,
    ).transform(dataframe)

    if is_spark:
        dataframe_with_sessions = dataframe_with_sessions.sort("user_id", "timestamp")
        user_ids = _get_column_list(dataframe_with_sessions, "user_id")
        item_ids = _get_column_list(dataframe_with_sessions, "item_id")
        session_ids = _get_column_list(dataframe_with_sessions, "session_id")
    else:
        dataframe_with_sessions.sort_values(["user_id", "timestamp"], inplace=True)
        user_ids = _get_column_list_pandas(dataframe_with_sessions, "user_id")
        item_ids = _get_column_list_pandas(dataframe_with_sessions, "item_id")
        session_ids = _get_column_list_pandas(dataframe_with_sessions, "session_id")

    assert user_ids == user_answer
    assert item_ids == item_answer
    assert len(set(session_ids)) == sessions_count


@pytest.mark.parametrize(
    "time_column, session_gap, min_sessions_per_user, max_sessions_per_user, user_answer, item_answer, sessions_count",
    [
        ("timestamp", 1, 6, 10, [], [], 0),
        ("timestamp", 1, 1, 5, [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3], [1, 2, 3, 1, 2, 3, 9, 1, 5, 3, 1, 2], 12),
        ("timestamp", 1, 4, 5, [2, 2, 2, 2, 3, 3, 3, 3, 3], [1, 2, 3, 9, 1, 5, 3, 1, 2], 9),
        ("spark_date", 86400, 6, 10, [], [], 0),
        ("spark_date", 43200, 1, 6, [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3], [1, 2, 3, 1, 2, 3, 9, 1, 5, 3, 1, 2], 12),
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("dataframe_sessionizer", marks=pytest.mark.spark),
        pytest.param("dataframe_sessionizer_pandas", marks=pytest.mark.core),
    ],
)
def test_sessionizer_sessions_per_user(
    time_column,
    session_gap,
    min_sessions_per_user,
    max_sessions_per_user,
    user_answer,
    item_answer,
    sessions_count,
    dataset,
    request,
):
    dataframe = request.getfixturevalue(dataset)
    is_spark = isinstance(dataframe, SparkDataFrame)
    time_column = "pandas_date" if is_spark is False and time_column == "spark_date" else time_column

    dataframe_with_sessions = Sessionizer(
        time_column=time_column,
        session_gap=session_gap,
        time_column_format="dd-MM-yyyy",
        min_sessions_per_user=min_sessions_per_user,
        max_sessions_per_user=max_sessions_per_user,
    ).transform(dataframe)
    if is_spark:
        dataframe_with_sessions = dataframe_with_sessions.sort("user_id", "timestamp")
        user_ids = _get_column_list(dataframe_with_sessions, "user_id")
        item_ids = _get_column_list(dataframe_with_sessions, "item_id")
        session_ids = _get_column_list(dataframe_with_sessions, "session_id")
    else:
        dataframe_with_sessions.sort_values(["user_id", "timestamp"], inplace=True)
        user_ids = _get_column_list_pandas(dataframe_with_sessions, "user_id")
        item_ids = _get_column_list_pandas(dataframe_with_sessions, "item_id")
        session_ids = _get_column_list_pandas(dataframe_with_sessions, "session_id")
    assert user_ids == user_answer
    assert item_ids == item_answer
    assert len(set(session_ids)) == sessions_count


@pytest.mark.parametrize(
    "session_gap, min_interactions, max_interactions, answer",
    [
        (30, None, None, [2, 1, 1, 6, 6, 6, 6, 8, 7]),
        (10, None, None, [2, 1, 0, 6, 5, 4, 3, 8, 7]),
        (30, 2, None, [1, 1, 6, 6, 6, 6]),
        (30, None, 3, [2, 1, 1, 8, 7]),
        (30, 4, 4, [6, 6, 6, 6]),
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("session_dataset_spark", marks=pytest.mark.spark),
        pytest.param("session_dataset_pandas", marks=pytest.mark.core),
    ],
)
def test_valid_session_ids(request, dataset, session_gap, min_interactions, max_interactions, answer):
    data = request.getfixturevalue(dataset)

    result = Sessionizer(
        time_column="timestamp",
        user_column="user_id",
        session_gap=session_gap,
        min_inter_per_session=min_interactions,
        max_inter_per_session=max_interactions,
        session_column="session_id",
    ).transform(data)

    if isinstance(result, SparkDataFrame):
        session_ids = _get_column_list(result.sort("user_id", "timestamp"), "session_id")
    else:
        session_ids = _get_column_list_pandas(result, "session_id")

    assert session_ids == answer
