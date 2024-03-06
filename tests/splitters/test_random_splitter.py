# pylint: disable-all
import numpy as np
import pandas as pd
import polars as pl
import pytest

from replay.splitters import RandomSplitter
from replay.utils import PYSPARK_AVAILABLE, SparkDataFrame
from tests.utils import spark

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf


@pytest.fixture()
def log():
    return pd.DataFrame(
        {
            "user_id": list(range(5000)),
            "item_id": list(range(5000)),
            "relevance": [1] * 5000,
        }
    )


@pytest.fixture()
@pytest.mark.usefixtures("spark")
def spark_dataframe_test(spark):
    columns = ["user_id", "item_id", "timestamp", "session_id"]
    data = [
        (1, 1, "01-01-2020", 1),
        (1, 2, "02-01-2020", 1),
        (1, 3, "03-01-2020", 1),
        (1, 4, "04-01-2020", 1),
        (1, 5, "05-01-2020", 1),
        (2, 1, "06-01-2020", 2),
        (2, 2, "07-01-2020", 2),
        (2, 3, "08-01-2020", 3),
        (2, 9, "09-01-2020", 4),
        (2, 10, "10-01-2020", 4),
        (3, 1, "01-01-2020", 5),
        (3, 5, "02-01-2020", 5),
        (3, 3, "03-01-2020", 5),
        (3, 1, "04-01-2020", 6),
        (3, 2, "05-01-2020", 6),
    ]
    return spark.createDataFrame(data, schema=columns).withColumn(
        "timestamp", sf.to_date("timestamp", "dd-MM-yyyy")
    )


@pytest.fixture(scope="module")
def pandas_dataframe_test():
    columns = ["user_id", "item_id", "timestamp", "session_id"]
    data = [
        (1, 1, "01-01-2020", 1),
        (1, 2, "02-01-2020", 1),
        (1, 3, "03-01-2020", 1),
        (1, 4, "04-01-2020", 1),
        (1, 5, "05-01-2020", 1),
        (2, 1, "06-01-2020", 2),
        (2, 2, "07-01-2020", 2),
        (2, 3, "08-01-2020", 3),
        (2, 9, "09-01-2020", 4),
        (2, 10, "10-01-2020", 4),
        (3, 1, "01-01-2020", 5),
        (3, 5, "02-01-2020", 5),
        (3, 3, "03-01-2020", 5),
        (3, 1, "04-01-2020", 6),
        (3, 2, "05-01-2020", 6),
    ]

    dataframe = pd.DataFrame(data, columns=columns)
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], format="%d-%m-%Y")

    return dataframe


@pytest.fixture()
def polars_dataframe_test(pandas_dataframe_test):
    return pl.from_pandas(pandas_dataframe_test)


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


SEED = 7777
test_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log_spark", marks=pytest.mark.spark),
        pytest.param("log", marks=pytest.mark.core),
        pytest.param("log_polars", marks=pytest.mark.core),
    ]
)
@pytest.mark.parametrize("test_size", test_sizes)
def test_nothing_is_lost(test_size, dataset_type, request):
    log = request.getfixturevalue(dataset_type)
    splitter = RandomSplitter(
        test_size=test_size,
        drop_cold_items=False,
        drop_cold_users=False,
        seed=SEED,
    )
    train, test = splitter.split(log)

    if not isinstance(log, SparkDataFrame):
        real_test_size = test.shape[0] / len(log)
        assert train.shape[0] + test.shape[0] == len(log)
    else:
        real_test_size = test.count() / log.count()
        assert train.count() + test.count() == log.count()

    assert np.isclose(real_test_size, test_size, atol=0.01)


def test_bad_test_size():
    with pytest.raises(ValueError):
        RandomSplitter(1.2)


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("spark_dataframe_test", marks=pytest.mark.spark),
        pytest.param("pandas_dataframe_test", marks=pytest.mark.core),
        pytest.param("polars_dataframe_test", marks=pytest.mark.core),
    ]
)
def test_with_session_ids(dataset_type, request):
    log = request.getfixturevalue(dataset_type)
    splitter = RandomSplitter(
        test_size=0.3,
        drop_cold_items=False,
        drop_cold_users=False,
        seed=SEED,
    )
    train, test = splitter.split(log)

    if not isinstance(log, SparkDataFrame):
        assert train.shape[0] + test.shape[0] == log.shape[0]
    else:
        assert train.count() + test.count() == log.count()


@pytest.mark.parametrize(
    "dataset_type",
    [
        pytest.param("log_spark", marks=pytest.mark.spark),
        pytest.param("log", marks=pytest.mark.core),
        pytest.param("log_polars", marks=pytest.mark.core),
    ]
)
def test_with_multiple_splitting(dataset_type, request):
    log = request.getfixturevalue(dataset_type)
    splitter = RandomSplitter(
        test_size=0.6,
        drop_cold_items=False,
        drop_cold_users=False,
        seed=SEED,
    )
    train, test = splitter.split(log)

    if not isinstance(log, SparkDataFrame):
        real_test_size = test.shape[0] / len(log)
        assert train.shape[0] + test.shape[0] == log.shape[0]
    else:
        real_test_size = test.count() / log.count()
        assert train.count() + test.count() == log.count()

    assert np.isclose(real_test_size, 0.6, atol=0.015)


@pytest.mark.core
def test_not_implemented_dataframe(log_not_implemented):
    with pytest.raises(NotImplementedError):
        RandomSplitter(0.2).split(log_not_implemented)
