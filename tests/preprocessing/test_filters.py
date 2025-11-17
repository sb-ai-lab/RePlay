from collections import Counter
from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from replay.preprocessing.filters import (
    ConsecutiveDuplicatesFilter,
    EntityDaysFilter,
    GlobalDaysFilter,
    LowRatingFilter,
    MinCountFilter,
    NumInteractionsFilter,
    QuantileItemsFilter,
    TimePeriodFilter,
    filter_cold,
)
from replay.utils import (
    PYSPARK_AVAILABLE,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
)
from replay.utils.session_handler import get_spark_session


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
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
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


def _make_dataframes(backend, spark=None):
    pd_target = pd.DataFrame({"user_id": [1, 1, 2, 3], "item_id": [10, 11, 12, 13]})
    pd_ref_items = pd.DataFrame({"item_id": [10, 12]})
    pd_ref_users = pd.DataFrame({"user_id": [1, 3]})

    if backend == "pandas":
        return pd_target, pd_ref_users, pd_ref_items
    if backend == "polars":
        return pl.from_pandas(pd_target), pl.from_pandas(pd_ref_users), pl.from_pandas(pd_ref_items)
    if backend == "spark":
        assert spark is not None
        return (
            spark.createDataFrame(pd_target),
            spark.createDataFrame(pd_ref_users),
            spark.createDataFrame(pd_ref_items),
        )
    msg = f"Unknown backend: {backend}"
    raise AssertionError(msg)


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


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("simple_data_to_filter_spark", marks=pytest.mark.spark),
        pytest.param("simple_data_to_filter_pandas", marks=pytest.mark.core),
        pytest.param("simple_data_to_filter_polars", marks=pytest.mark.core),
    ],
)
def test_quantile_items_filter(dataset_type, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = QuantileItemsFilter(0.7).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        users_init = test_dataframe["query_id"].tolist()
        users_filtered = filtered_dataframe["query_id"].tolist()
        items_init = set(test_dataframe["item_id"].unique().tolist())
        items_filtered = set(filtered_dataframe["item_id"].unique().tolist())
        items_distribution_init = test_dataframe.groupby("item_id").size()
        items_distribution_filtered = filtered_dataframe.groupby("item_id").size()
    elif isinstance(test_dataframe, PolarsDataFrame):
        users_init = test_dataframe["query_id"].to_list()
        users_filtered = filtered_dataframe["query_id"].to_list()
        items_init = set(test_dataframe["item_id"].unique().to_list())
        items_filtered = set(filtered_dataframe["item_id"].unique().to_list())
        items_distribution_init_df = test_dataframe.group_by("item_id").count()
        items_distribution_filtered_df = filtered_dataframe.group_by("item_id").count()
        items_distribution_init = dict(
            zip(items_distribution_init_df["item_id"].to_list(), items_distribution_init_df["count"].to_list())
        )
        items_distribution_filtered = dict(
            zip(items_distribution_filtered_df["item_id"].to_list(), items_distribution_filtered_df["count"].to_list())
        )
    else:
        users_init = [x.query_id for x in test_dataframe.select("query_id").collect()]
        users_filtered = [x.query_id for x in filtered_dataframe.select("query_id").collect()]
        items_init = {x.item_id for x in test_dataframe.select("item_id").distinct().collect()}
        items_filtered = {x.item_id for x in filtered_dataframe.select("item_id").distinct().collect()}
        items_distribution_init_df = test_dataframe.groupBy("item_id").count()
        items_distribution_filtered_df = filtered_dataframe.groupBy("item_id").count()
        items_distribution_init = dict(
            zip(
                [x.item_id for x in items_distribution_init_df.select("item_id").collect()],
                [x["count"] for x in items_distribution_init_df.select("count").collect()],
            )
        )
        items_distribution_filtered = dict(
            zip(
                [x.item_id for x in items_distribution_filtered_df.select("item_id").collect()],
                [x["count"] for x in items_distribution_filtered_df.select("count").collect()],
            )
        )

    users_init = Counter(users_init)
    users_filtered = Counter(users_filtered)
    assert users_init[0] == users_filtered[0] + 3
    for key in [1, 2, 3, 4, 5]:
        assert users_init[key] == users_filtered[key]

    assert items_init == items_filtered
    assert items_distribution_init[12] == items_distribution_filtered[12]
    assert items_distribution_init[13] == items_distribution_filtered[13]
    assert items_distribution_init[14] == items_distribution_filtered[14]
    assert items_distribution_init[15] == items_distribution_filtered[15]
    assert items_distribution_init[10] == items_distribution_filtered[10] + 1
    assert items_distribution_init[11] == items_distribution_filtered[11] + 2


@pytest.fixture(scope="module")
def consecutive_duplicates_with_signle_item():
    inputs = pd.DataFrame(
        {"query_id": ["u1", "u2", "u1", "u2", "u2", "u3"], "item_id": ["i1"] * 6, "timestamp": list(range(6))}
    )
    target = pd.DataFrame({"query_id": ["u1", "u2", "u3"], "item_id": ["i1"] * 3, "timestamp": [0, 1, 5]})
    return inputs, target, "first"


@pytest.fixture(scope="module")
def consecutive_duplicates_with_extra_column():
    inputs = pd.DataFrame(
        {
            "query_id": ["u3", "u1", "u2", "u1", "u1", "u0"],
            "item_id": ["i2", "i1", "i1", "i1", "i2", "i2"],
            "rating": [2, 1, 3, 4, 1, 5],
            "timestamp": list(range(6)),
        }
    )
    target = pd.DataFrame(
        {
            "query_id": ["u0", "u1", "u1", "u2", "u3"],
            "item_id": ["i2", "i1", "i2", "i1", "i2"],
            "rating": [5, 1, 1, 3, 2],
            "timestamp": [5, 1, 4, 2, 0],
        }
    )
    return inputs, target, "first"


@pytest.fixture(scope="module")
def consecutive_duplicates_for_testing_keep_param():
    target = pd.DataFrame(
        {
            "query_id": ["u0", "u1", "u2", "u3", "u4"],
            "item_id": ["i0", "i1", "i1", "i2", "i0"],
            "timestamp": pd.date_range("2024-01-01", "2024-01-05"),
        }
    )
    duplicates = target.copy()
    duplicates["timestamp"] += pd.Timedelta(days=-1)
    inputs = pd.concat([target, duplicates])
    return inputs, target, "last"


def to_backend(dataframe, backend):
    if backend == "polars":
        return pl.from_pandas(dataframe)
    elif backend == "spark":
        return get_spark_session().createDataFrame(dataframe)
    else:
        return dataframe


def to_pandas(dataframe):
    if isinstance(dataframe, PolarsDataFrame):
        return dataframe.to_pandas()
    elif isinstance(dataframe, SparkDataFrame):
        return dataframe.toPandas()
    else:
        return dataframe


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("pandas", marks=pytest.mark.core),
        pytest.param("polars", marks=pytest.mark.core),
        pytest.param("spark", marks=pytest.mark.spark),
    ],
)
@pytest.mark.parametrize(
    "consecutive_duplicates",
    [
        "consecutive_duplicates_with_signle_item",
        "consecutive_duplicates_with_extra_column",
        "consecutive_duplicates_for_testing_keep_param",
    ],
)
def test_consecutive_duplicates_filter(consecutive_duplicates, backend, request):
    inputs, target, keep = request.getfixturevalue(consecutive_duplicates)
    inputs = to_backend(inputs, backend)
    filtered = to_pandas(ConsecutiveDuplicatesFilter(keep=keep).transform(inputs))
    assert (filtered.sort_values("query_id").reset_index(drop=True) == target).all(axis=None)


@pytest.mark.core
def test_consecutive_duplicates_filter_invalid_param_keep_error():
    with pytest.raises(ValueError, match=r"`keep` must be either 'first' or 'last'"):
        ConsecutiveDuplicatesFilter(keep="42")


@pytest.mark.core
@pytest.mark.parametrize("quantile, items_proportion", [(2, 0.5), (0.5, -1)])
def test_quantile_filter_error(quantile, items_proportion):
    with pytest.raises(ValueError):
        QuantileItemsFilter(quantile, items_proportion=items_proportion)


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
        (ConsecutiveDuplicatesFilter, {}),
    ],
)
def test_filter_not_implemented(filter, kwargs, interactions_not_implemented):
    with pytest.raises(NotImplementedError):
        filter(**kwargs).transform(interactions_not_implemented)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("pandas", marks=pytest.mark.core),
        pytest.param("polars", marks=pytest.mark.core),
        pytest.param("spark", marks=pytest.mark.spark),
    ],
)
def test_filter_cold_items(backend, request, spark=None):
    if backend == "spark" and not PYSPARK_AVAILABLE:
        pytest.skip("PySpark is not available")
    if backend == "spark":
        spark = request.getfixturevalue("spark")

    target, _, ref_items = _make_dataframes(backend, spark)

    result = filter_cold(
        target=target,
        reference=ref_items,
        mode="items",
        query_column="user_id",
        item_column="item_id",
    )

    if isinstance(result, PandasDataFrame):
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), pd.DataFrame({"user_id": [1, 2], "item_id": [10, 12]})
        )
    elif isinstance(result, PolarsDataFrame):
        assert result.equals(pl.DataFrame({"user_id": [1, 2], "item_id": [10, 12]}))
    else:
        rows = {(r.user_id, r.item_id) for r in result.collect()}
        assert rows == {(1, 10), (2, 12)}


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("pandas", marks=pytest.mark.core),
        pytest.param("polars", marks=pytest.mark.core),
        pytest.param("spark", marks=pytest.mark.spark),
    ],
)
def test_filter_cold_users(backend, request, spark=None):
    if backend == "spark" and not PYSPARK_AVAILABLE:
        pytest.skip("PySpark is not available")
    if backend == "spark":
        spark = request.getfixturevalue("spark")

    target, ref_users, _ = _make_dataframes(backend, spark)

    result = filter_cold(
        target=target,
        reference=ref_users,
        mode="users",
        query_column="user_id",
        item_column="item_id",
    )

    if isinstance(result, PandasDataFrame):
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), pd.DataFrame({"user_id": [1, 1, 3], "item_id": [10, 11, 13]})
        )
    elif isinstance(result, PolarsDataFrame):
        assert result.equals(pl.DataFrame({"user_id": [1, 1, 3], "item_id": [10, 11, 13]}))
    else:
        rows = {(r.user_id, r.item_id) for r in result.collect()}
        assert rows == {(1, 10), (1, 11), (3, 13)}


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("pandas", marks=pytest.mark.core),
        pytest.param("polars", marks=pytest.mark.core),
        pytest.param("spark", marks=pytest.mark.spark),
    ],
)
def test_filter_cold_both(backend, request, spark=None):
    if backend == "spark" and not PYSPARK_AVAILABLE:
        pytest.skip("PySpark is not available")
    if backend == "spark":
        spark = request.getfixturevalue("spark")

    target, _, _ = _make_dataframes(backend, spark)

    if backend == "pandas":
        ref_both = pd.DataFrame({"user_id": [1, 3], "item_id": [10, 12]})
    elif backend == "polars":
        ref_both = pl.DataFrame({"user_id": [1, 3], "item_id": [10, 12]})
    else:
        ref_both = spark.createDataFrame(pd.DataFrame({"user_id": [1, 3], "item_id": [10, 12]}))

    result = filter_cold(
        target=target,
        reference=ref_both,
        mode="both",
        query_column="user_id",
        item_column="item_id",
    )

    if isinstance(result, PandasDataFrame):
        pd.testing.assert_frame_equal(result.reset_index(drop=True), pd.DataFrame({"user_id": [1], "item_id": [10]}))
    elif isinstance(result, PolarsDataFrame):
        assert result.equals(pl.DataFrame({"user_id": [1], "item_id": [10]}))
    else:
        rows = {(r.user_id, r.item_id) for r in result.collect()}
        assert rows == {(1, 10)}


@pytest.mark.core
def test_filter_cold_invalid_mode_raises():
    target = pd.DataFrame({"user_id": [1], "item_id": [10]})
    ref = pd.DataFrame({"user_id": [1]})
    msg = "mode must be 'items' | 'users' | 'both'"
    with pytest.raises(ValueError, match=msg):
        filter_cold(target, ref, mode="invalid")


@pytest.mark.core
def test_filter_cold_type_mismatch_raises():
    target = pd.DataFrame({"user_id": [1], "item_id": [10]})
    ref = pl.DataFrame({"user_id": [1], "item_id": [10]})
    with pytest.raises(TypeError, match="Target and reference must be of the same type"):
        filter_cold(target, ref, mode="items")


@pytest.mark.core
def test_filter_cold_missing_column_raises():
    target = pd.DataFrame({"user_id": [1], "item_id": [10]})
    ref_missing_item = pd.DataFrame({"user_id": [1]})
    with pytest.raises(KeyError, match="Column 'item_id' must be in both dataframes"):
        filter_cold(target, ref_missing_item, mode="items")


class _DummyDF:
    def __init__(self, columns):
        self.columns = columns


@pytest.mark.core
def test_filter_cold_unsupported_type_raises_notimplemented():
    target = _DummyDF(["user_id", "item_id"])
    ref = _DummyDF(["user_id", "item_id"])
    with pytest.raises(NotImplementedError, match="Unsupported data frame type"):
        filter_cold(target, ref, mode="items")
