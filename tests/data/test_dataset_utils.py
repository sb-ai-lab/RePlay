import pytest

from replay.data.dataset import nunique, select
from replay.utils import SparkDataFrame
from tests.utils import spark


@pytest.mark.spark
@pytest.mark.parametrize(
    "dataset, column, number_of_unique",
    [
        ("full_spark_dataset", "user_id", 3),
        ("full_pandas_dataset", "user_id", 3),
        ("full_spark_dataset", "item_id", 4),
        ("full_pandas_dataset", "item_id", 4),
    ],
)
def test_number_of_unique_values(dataset, column, number_of_unique, request):
    dataset = request.getfixturevalue(dataset)["interactions"]

    assert nunique(dataset, column) == number_of_unique


@pytest.mark.parametrize(
    "dataset, columns, shape",
    [
        pytest.param("full_spark_dataset", ["user_id"], (6, 1), marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", ["user_id"], (6, 1), marks=pytest.mark.core),
        pytest.param("full_spark_dataset", ["user_id", "item_id"], (6, 2), marks=pytest.mark.spark),
        pytest.param("full_pandas_dataset", ["user_id", "item_id"], (6, 2), marks=pytest.mark.core),
    ],
)
def test_column_selection(dataset, columns, shape, request):
    dataset = request.getfixturevalue(dataset)["interactions"]
    selected = select(dataset, columns)

    if isinstance(dataset, SparkDataFrame):
        assert (selected.count(), len(selected.columns)) == shape
    else:
        assert selected.shape == shape


@pytest.mark.core
def test_assertion_in_select():
    with pytest.raises(AssertionError) as exc:
        select("string", ["user_id"])

    assert str(exc.value) == "Unknown data frame type"
