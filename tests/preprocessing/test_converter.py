import pytest

from replay.preprocessing import CSRConverter
from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf


@pytest.fixture(scope="module")
def interactions_pandas():
    return PandasDataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            "item_id": [3, 7, 10, 5, 8, 11, 4, 9, 2, 5],
            "rating": [1, 2, 3, 3, 2, 1, 3, 12, 1, 4],
        }
    )


@pytest.mark.usefixtures("spark")
@pytest.fixture()
def interactions_spark(spark, interactions_pandas):
    return spark.createDataFrame(interactions_pandas)


@pytest.fixture(scope="module")
def true_size(interactions_pandas):
    user_cnt = interactions_pandas["user_id"].max()
    item_cnt = interactions_pandas["item_id"].max()
    return (user_cnt + 1, item_cnt + 1)


@pytest.mark.spark
@pytest.mark.parametrize("row_count", [None, 1000, 1500])
@pytest.mark.parametrize("column_count", [None, 2000, 1700])
@pytest.mark.usefixtures("interactions_spark", "true_size")
def test_CSRConverter_user_column_counts(row_count, column_count, interactions_spark, true_size):
    current_size = (
        row_count if row_count is not None else true_size[0],
        column_count if column_count is not None else true_size[1],
    )
    csr = CSRConverter(
        first_dim_column="user_id", second_dim_column="item_id", row_count=row_count, column_count=column_count
    ).transform(interactions_spark)
    assert csr.shape == current_size


@pytest.mark.spark
@pytest.mark.parametrize("row_count", [3, 2, 1])
@pytest.mark.parametrize("column_count", [11, 1, 5])
@pytest.mark.usefixtures("interactions_spark")
def test_CSRConverter_user_column_counts_exception(row_count, column_count, interactions_spark):
    with pytest.raises(ValueError):
        CSRConverter(
            first_dim_column="user_id", second_dim_column="item_id", row_count=row_count, column_count=column_count
        ).transform(interactions_spark)


@pytest.mark.parametrize("data_column", [None, "rating"])
@pytest.mark.parametrize(
    "data",
    [
        pytest.param("interactions_spark", marks=pytest.mark.spark),
        pytest.param("interactions_pandas", marks=pytest.mark.core),
    ],
)
def test_CSRConverter_rating_column(data_column, data, request):
    data = request.getfixturevalue(data)
    csr = CSRConverter(first_dim_column="user_id", second_dim_column="item_id", data_column=data_column).transform(data)
    if data_column is None:
        answer = data.shape[0] if isinstance(data, PandasDataFrame) else data.count()
    else:
        answer = data["rating"].sum() if isinstance(data, PandasDataFrame) else data.select(sf.sum("rating")).first()[0]
    assert csr.todense().sum() == answer
