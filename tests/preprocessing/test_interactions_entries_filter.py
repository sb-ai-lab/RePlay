import pytest

from replay.preprocessing.filters import InteractionEntriesFilter
from replay.utils import PandasDataFrame


@pytest.fixture
def interactions_pandas():
    columns = ["user_id", "item_id"]
    data = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 3), (3, 4), (4, 1), (4, 3), (4, 4)]
    return PandasDataFrame(data, columns=columns)


@pytest.fixture
@pytest.mark.usefixtures("spark")
def interactions_spark(spark, interactions_pandas):
    return spark.createDataFrame(interactions_pandas)


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_by_min_users(dataset_type, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = InteractionEntriesFilter(min_inter_per_user=3).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        user_list = filtered_dataframe["user_id"].unique().tolist()
        assert len(user_list) == 2
        assert sorted(user_list)[0] == 3
    else:
        user_list = filtered_dataframe.select("user_id").distinct().collect()
        ids = [x[0] for x in user_list]
        assert len(ids) == 2
        assert sorted(ids)[0] == 3


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_by_max_users(dataset_type, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = InteractionEntriesFilter(max_inter_per_user=1).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        user_list = filtered_dataframe["user_id"].unique().tolist()
        assert len(user_list) == 1
        assert user_list[0] == 1
    else:
        user_list = filtered_dataframe.select("user_id").distinct().collect()
        assert len([x[0] for x in user_list]) == 1
        assert user_list[0][0] == 1


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_by_min_items(dataset_type, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = InteractionEntriesFilter(min_inter_per_item=3).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        user_list = filtered_dataframe["item_id"].unique().tolist()
        assert len(user_list) == 1
        assert user_list[0] == 1
    else:
        user_list = filtered_dataframe.select("item_id").distinct().collect()
        assert len([x[0] for x in user_list]) == 1
        assert user_list[0][0] == 1


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_by_max_items(dataset_type, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = InteractionEntriesFilter(max_inter_per_item=1).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        item_list = filtered_dataframe["item_id"].unique().tolist()
        assert len(item_list) == 1
        assert item_list[0] == 2
    else:
        item_list = filtered_dataframe.select("item_id").distinct().collect()
        assert len([x[0] for x in item_list]) == 1
        assert item_list[0][0] == 2


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_by_min_max_users(dataset_type, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = InteractionEntriesFilter(min_inter_per_user=1, max_inter_per_user=2).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        user_list = filtered_dataframe["user_id"].unique().tolist()
        assert len(user_list) == 2
        assert sorted(user_list)[0] == 1
    else:
        user_list = filtered_dataframe.select("user_id").distinct().collect()
        idx = [x[0] for x in user_list]
        assert len(idx) == 2
        assert sorted(idx)[0] == 1


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_by_min_max_items(dataset_type, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = InteractionEntriesFilter(min_inter_per_item=2, max_inter_per_item=3).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        item_list = filtered_dataframe["item_id"].unique().tolist()
        assert len(item_list) == 2
        assert sorted(item_list)[0] == 3
    else:
        item_list = filtered_dataframe.select("item_id").distinct().collect()
        idx = [x[0] for x in item_list]
        assert len(idx) == 2
        assert sorted(idx)[0] == 3


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
    ],
)
def test_interaction_entries_filter_by_min_max_users_items(dataset_type, request):
    test_dataframe = request.getfixturevalue(dataset_type)
    filtered_dataframe = InteractionEntriesFilter(
        min_inter_per_user=2,
        min_inter_per_item=2,
    ).transform(test_dataframe)

    if isinstance(test_dataframe, PandasDataFrame):
        user_list = filtered_dataframe["user_id"].unique().tolist()
        item_list = filtered_dataframe["item_id"].unique().tolist()
    else:
        user_list = [user_id[0] for user_id in filtered_dataframe.select("user_id").distinct().collect()]
        item_list = [user_id[0] for user_id in filtered_dataframe.select("item_id").distinct().collect()]

    assert set(user_list) == set([3, 4])
    assert set(item_list) == set([1, 3, 4])
