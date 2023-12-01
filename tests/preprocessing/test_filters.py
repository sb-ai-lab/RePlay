import pytest
import pandas as pd
from datetime import datetime
from pandas import DataFrame as PandasDataFrame

from replay.utils import get_spark_session
from replay.preprocessing.filters import (
    MinCountFilter,
    LowRatingFilter,
    NumInteractionsFilter,
    EntityDaysFilter,
    GlobalDaysFilter,
    TimePeriodFilter,
)


@pytest.fixture
def interactions_pandas():
    df = PandasDataFrame(
        {
            "user_id": ["u1", "u2", "u2", "u3", "u3", "u3"],
            "item_id": ["i1", "i2","i3", "i1", "i2","i3"],
            "rating": [1., 0.5, 3, 1, 0, 1],
            "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
                          "2020-02-01", "2020-01-01 00:04:15",
                          "2020-01-02 00:04:14", "2020-01-05 23:59:59"]
        },
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def interactions_spark(interactions_pandas):
    return get_spark_session().createDataFrame(interactions_pandas)


@pytest.mark.spark
@pytest.mark.parametrize(
    "dataset_type",
    [
        ("interactions_spark"),
        ("interactions_pandas"),
    ],
)
def test_mincount_filter(dataset_type, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = MinCountFilter(num_entries=3).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        user_list = filtered_dataframe["user_id"].unique().tolist()
        assert len(user_list) == 1
        assert sorted(user_list)[0] == "u3"
    else:
        user_list = filtered_dataframe.select("user_id").distinct().collect()
        ids = [x[0] for x in user_list]
        assert len(ids) == 1
        assert sorted(ids)[0] == "u3"


@pytest.mark.spark
@pytest.mark.parametrize(
    "dataset_type",
    [
        ("interactions_spark"),
        ("interactions_pandas"),
    ],
)
def test_lowrating_filter(dataset_type, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = LowRatingFilter(value=3).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        user_list = filtered_dataframe["user_id"].unique().tolist()
        assert len(user_list) == 1
        assert sorted(user_list)[0] == "u2"
    else:
        user_list = filtered_dataframe.select("user_id").distinct().collect()
        ids = [x[0] for x in user_list]
        assert len(ids) == 1
        assert sorted(ids)[0] == "u2"


@pytest.mark.spark
@pytest.mark.parametrize(
    "dataset_type",
    [
        ("interactions_spark"),
        ("interactions_pandas"),
    ],
)
@pytest.mark.parametrize(
    "first", [(True), (False)],
)
@pytest.mark.parametrize(
    "item_column", [(None), ("item_id")],
)
def test_numinteractions_filter(dataset_type, first, item_column, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = NumInteractionsFilter(
        num_interactions=1, first=first, item_column=item_column
    ).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        user_list = filtered_dataframe["user_id"].unique().tolist()
        assert len(user_list) == 3
    else:
        user_list = filtered_dataframe.select("user_id").distinct().collect()
        ids = [x[0] for x in user_list]
        assert len(ids) == 3


@pytest.mark.spark
@pytest.mark.parametrize(
    "dataset_type",
    [
        ("interactions_spark"),
        ("interactions_pandas"),
    ],
)
@pytest.mark.parametrize(
    "first, answer", [(True, 5), (False, 4)],
)
def test_entitydays_filter(dataset_type, first, answer, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = EntityDaysFilter(
        days=1, first=first,
    ).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        assert len(filtered_dataframe) == answer
    else:
        assert filtered_dataframe.count() == answer


@pytest.mark.spark
@pytest.mark.parametrize(
    "dataset_type",
    [
        ("interactions_spark"),
        ("interactions_pandas"),
    ],
)
@pytest.mark.parametrize(
    "first, answer", [(True, 3), (False, 2)],
)
def test_globaldays_filter(dataset_type, first, answer, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = GlobalDaysFilter(
        days=1, first=first,
    ).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        item_list = filtered_dataframe["item_id"].unique().tolist()
        assert len(filtered_dataframe) == answer
        assert len(item_list) == 2
    else:
        item_list = filtered_dataframe.select("item_id").distinct().collect()
        ids = [x[0] for x in item_list]
        assert filtered_dataframe.count() == answer
        assert len(ids) == 2


@pytest.mark.spark
@pytest.mark.parametrize(
    "dataset_type",
    [
        ("interactions_spark"),
        ("interactions_pandas"),
    ],
)
@pytest.mark.parametrize(
    "start, end, answer, item_ids",
    [
        ("2020-01-01 14:00:00", datetime(2020, 1, 3, 0, 0, 0), 2, 2),
        (None, None, 6, 3),
    ],
)
def test_timeperiod_filter(dataset_type, start, end, answer, item_ids, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = TimePeriodFilter(
        start_date=start, end_date=end
    ).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        item_list = filtered_dataframe["item_id"].unique().tolist()
        assert len(filtered_dataframe) == answer
        assert len(item_list) == item_ids
    else:
        item_list = filtered_dataframe.select("item_id").distinct().collect()
        ids = [x[0] for x in item_list]
        assert filtered_dataframe.count() == answer
        assert len(ids) == item_ids
