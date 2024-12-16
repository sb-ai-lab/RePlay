import pytest

from replay.data.nn.utils import ensure_pandas, groupby_sequences
from replay.utils import PandasDataFrame, get_spark_session


@pytest.mark.parametrize(
    "is_spark, sort_col",
    [
        pytest.param(False, None, marks=pytest.mark.core),
        pytest.param(False, "timestamp", marks=pytest.mark.core),
        pytest.param(True, None, marks=pytest.mark.spark),
        pytest.param(True, "timestamp", marks=pytest.mark.spark),
    ],
)
def test_groupby_sequences_pandas(pandas_interactions, is_spark, sort_col):
    if is_spark is True:
        grouped = (
            groupby_sequences(
                get_spark_session().createDataFrame(pandas_interactions), groupby_col="user_id", sort_col=sort_col
            )
            .toPandas()
            .sort_values("user_id")
        )
    else:
        grouped = groupby_sequences(pandas_interactions, groupby_col="user_id", sort_col=sort_col).sort_values(
            "user_id"
        )

    assert list(grouped["item_id"].iloc[0]) == [1, 2]
    assert list(grouped["item_id"].iloc[-1]) == [1, 2, 3, 4, 5, 6]
    assert list(grouped["timestamp"].iloc[0]) == [0, 1]
    assert list(grouped["timestamp"].iloc[-1]) == [6, 7, 8, 9, 10, 11]


@pytest.mark.spark
def test_ensure_pandas(pandas_interactions):
    assert isinstance(ensure_pandas(get_spark_session().createDataFrame(pandas_interactions)), PandasDataFrame)
    assert isinstance(ensure_pandas(pandas_interactions), PandasDataFrame)
