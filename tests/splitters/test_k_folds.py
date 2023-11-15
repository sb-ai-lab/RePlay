# pylint: disable-all
import pandas as pd
import pytest

from replay.splitters import KFolds
from tests.utils import spark


@pytest.fixture
def df():
    df = pd.DataFrame(
        {"user_id": [1, 1, 1, 2, 2], "item_id": [1, 2, 3, 4, 5],
         "session_id": [1, 1, 2, 1, 1], "timestamp": [1, 2, 3, 2, 3]}
    )
    return df


@pytest.fixture
def df_spark(spark, df):
    return spark.createDataFrame(df)


@pytest.mark.spark
def test_sum_spark(df_spark):
    res = pd.DataFrame()
    cv = KFolds(n_folds=2, seed=1337, session_id_column="session_id", query_column="user_id")
    for _, test in cv.split(df_spark):
        res = res.append(test.toPandas(), ignore_index=True)
    res = res.sort_values(["user_id", "item_id"]).reset_index(drop=True)
    assert all(res == df_spark)


@pytest.mark.core
def test_sum_pandas(df):
    res = pd.DataFrame()
    cv = KFolds(n_folds=2, seed=1337, session_id_column="session_id", query_column="user_id")
    for _, test in cv.split(df):
        res = res.append(test, ignore_index=True)
    res = res.sort_values(["user_id", "item_id"]).reset_index(drop=True)
    assert all(res == df)


@pytest.mark.core
def test_wrong_type():
    with pytest.raises(ValueError):
        next(KFolds(2, strategy="totally not query"))
