from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from replay.preprocessing.filters import (
    EntityDaysFilter,
    GlobalDaysFilter,
    LowRatingFilter,
    MinCountFilter,
    NumInteractionsFilter,
    TimePeriodFilter,
)
from replay.utils import PandasDataFrame, PolarsDataFrame, get_spark_session


@pytest.fixture
def interactions_pandas():
    df = PandasDataFrame(
        {
            "user_id": ["u1", "u2", "u2", "u3", "u3", "u3"],
            "item_id": ["i1", "i2", "i3", "i1", "i2", "i3"],
            "rating": [1.0, 0.5, 3, 1, 0, 1],
            "timestamp": [
                "2020-01-01 23:59:59",
                "2020-02-01",
                "2020-02-01",
                "2020-01-01 00:04:15",
                "2020-01-02 00:04:14",
                "2020-01-05 23:59:59",
            ],
        },
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def interactions_spark(interactions_pandas):
    return get_spark_session().createDataFrame(interactions_pandas)


@pytest.fixture
def interactions_polars(interactions_pandas):
    return pl.from_pandas(interactions_pandas)


@pytest.fixture
def interactions_not_implemented(interactions_pandas):
    return interactions_pandas.to_numpy()


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
        pytest.param("interactions_polars", marks=pytest.mark.core),
    ],
)
def test_mincount_filter(dataset_type, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = MinCountFilter(num_entries=3).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        user_list = filtered_dataframe["user_id"].unique().tolist()
    elif isinstance(test_dataframe, PolarsDataFrame):
        user_list = filtered_dataframe["user_id"].unique().to_list()
    else:
        user_list_df = filtered_dataframe.select("user_id").distinct().collect()
        user_list = [x[0] for x in user_list_df]

    assert len(user_list) == 1
    assert user_list[0] == "u3"


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
        pytest.param("interactions_polars", marks=pytest.mark.core),
    ],
)
def test_lowrating_filter(dataset_type, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = LowRatingFilter(value=3).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        user_list = filtered_dataframe["user_id"].unique().tolist()
    elif isinstance(test_dataframe, PolarsDataFrame):
        user_list = filtered_dataframe["user_id"].unique().to_list()
    else:
        user_list_df = filtered_dataframe.select("user_id").distinct().collect()
        user_list = [x[0] for x in user_list_df]

    assert len(user_list) == 1
    assert user_list[0] == "u2"


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
        pytest.param("interactions_polars", marks=pytest.mark.core),
    ],
)
@pytest.mark.parametrize(
    "first",
    [(True), (False)],
)
@pytest.mark.parametrize(
    "item_column",
    [(None), ("item_id")],
)
def test_numinteractions_filter(dataset_type, first, item_column, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = NumInteractionsFilter(num_interactions=2, first=first, item_column=item_column).transform(
        test_dataframe
    )

    if isinstance(test_dataframe, PandasDataFrame):
        user_list = filtered_dataframe["user_id"].unique().tolist()
    elif isinstance(test_dataframe, PolarsDataFrame):
        user_list = filtered_dataframe["user_id"].unique().to_list()
    else:
        user_list_df = filtered_dataframe.select("user_id").distinct().collect()
        user_list = [x[0] for x in user_list_df]

    assert len(user_list) == 3


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
        pytest.param("interactions_polars", marks=pytest.mark.core),
    ],
)
@pytest.mark.parametrize(
    "first, answer",
    [(True, 5), (False, 4)],
)
def test_entitydays_filter(dataset_type, first, answer, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = EntityDaysFilter(
        days=1,
        first=first,
    ).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        assert len(filtered_dataframe) == answer
    elif isinstance(test_dataframe, PolarsDataFrame):
        assert len(filtered_dataframe) == answer
    else:
        assert filtered_dataframe.count() == answer


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
        pytest.param("interactions_polars", marks=pytest.mark.core),
    ],
)
@pytest.mark.parametrize(
    "first, answer",
    [(True, 3), (False, 2)],
)
def test_globaldays_filter(dataset_type, first, answer, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = GlobalDaysFilter(
        days=1,
        first=first,
    ).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        item_list = filtered_dataframe["item_id"].unique().tolist()
        items_count = len(filtered_dataframe)
    elif isinstance(test_dataframe, PolarsDataFrame):
        item_list = filtered_dataframe["item_id"].unique().to_list()
        items_count = len(filtered_dataframe)
    else:
        item_list_df = filtered_dataframe.select("item_id").distinct().collect()
        item_list = [x[0] for x in item_list_df]
        items_count = filtered_dataframe.count()

    assert items_count == answer
    assert len(item_list) == 2


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
        pytest.param("interactions_polars", marks=pytest.mark.core),
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
    filtered_dataframe = TimePeriodFilter(start_date=start, end_date=end).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        item_list = filtered_dataframe["item_id"].unique().tolist()
        items_count = len(filtered_dataframe)
    elif isinstance(test_dataframe, PolarsDataFrame):
        item_list = filtered_dataframe["item_id"].unique().to_list()
        items_count = len(filtered_dataframe)
    else:
        item_list_df = filtered_dataframe.select("item_id").distinct().collect()
        item_list = [x[0] for x in item_list_df]
        items_count = filtered_dataframe.count()

    assert items_count == answer
    assert len(item_list) == item_ids


@pytest.mark.core
@pytest.mark.parametrize(
    "filter, kwargs",
    [
        (MinCountFilter, {"num_entries": 1}),
        (LowRatingFilter, {"value": 3}),
        (NumInteractionsFilter, {}),
        (EntityDaysFilter, {}),
        (GlobalDaysFilter, {}),
        (TimePeriodFilter, {}),
    ],
)
def test_filter_not_implemented(filter, kwargs, interactions_not_implemented):
    with pytest.raises(NotImplementedError):
        filter(**kwargs).transform(interactions_not_implemented)
