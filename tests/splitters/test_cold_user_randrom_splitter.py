# pylint: disable-all
import numpy as np
import pandas as pd
import polars as pl
import pytest

from replay.splitters import ColdUserRandomSplitter
from tests.utils import spark


@pytest.fixture()
def log():
    return pd.DataFrame(
        {
            "user_id": list(range(5000)),
            "item_id": list(range(5000)),
            "relevance": [1] * 5000,
            "timestamp": [1] * 5000,
        }
    )


@pytest.fixture()
@pytest.mark.usefixtures("spark")
def log_spark(spark, log):
    return spark.createDataFrame(log)


@pytest.fixture()
def log_polars(log):
    return pl.from_pandas(log)


@pytest.fixture()
def log_not_implemented(log):
    return log.to_numpy()


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log_spark", marks=pytest.mark.spark),
        pytest.param("log", marks=pytest.mark.core),
        pytest.param("log_polars", marks=pytest.mark.core),
    ]
)
def test_splitting(dataset_type, request):
    ratio = 0.25
    log = request.getfixturevalue(dataset_type)
    cold_user_splitter = ColdUserRandomSplitter(ratio, query_column="user_id")
    cold_user_splitter.seed = 27
    train, test = cold_user_splitter.split(log)

    if isinstance(log, pd.DataFrame):
        test_users = test.user_id.unique()
        train_users = train.user_id.unique()
        real_ratio = len(test_users) / len(log)
    elif isinstance(log, pl.DataFrame):
        test_users = test.select("user_id").unique()
        train_users = train.select("user_id").unique()
        real_ratio = len(test_users) / len(log)
    else:
        test_users = test.toPandas().user_id.unique()
        train_users = train.toPandas().user_id.unique()
        real_ratio = len(test_users) / log.count()

    assert not np.isin(test_users, train_users).any()
    assert np.isclose(
        real_ratio, ratio, atol=0.01
    )  # Spark weights are random ¯\_(ツ)_/¯


@pytest.mark.core
def test_invalid_test_size():
    with pytest.raises(ValueError):
        ColdUserRandomSplitter(test_size=1.2, query_column="user_id")


@pytest.mark.core
def test_not_implemented_dataframe(log_not_implemented):
    with pytest.raises(NotImplementedError):
        ColdUserRandomSplitter(test_size=0.2).split(log_not_implemented)
