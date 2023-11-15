# pylint: disable-all
import numpy as np
import pandas as pd
import pytest

from replay.splitters import RandomSplitter
from replay.utils import PYSPARK_AVAILABLE, get_spark_session

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf


@pytest.fixture
def log():
    return pd.DataFrame(
        {
            "user_id": list(range(5000)),
            "item_id": list(range(5000)),
            "relevance": [1] * 5000,
        }
    )


@pytest.fixture(scope="module")
def spark_dataframe_test():
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
    return get_spark_session().createDataFrame(data, schema=columns).withColumn(
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


@pytest.fixture
def log_spark(log):
    return get_spark_session().createDataFrame(log)


SEED = 7777
test_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]


@pytest.mark.parametrize(
    "dataset_type",
    [
        ("log"),
        ("log_spark"),
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

    if isinstance(log, pd.DataFrame):
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
        ("spark_dataframe_test"),
        ("pandas_dataframe_test"),
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

    if isinstance(log, pd.DataFrame):
        assert train.shape[0] + test.shape[0] == log.shape[0]
    else:
        assert train.count() + test.count() == log.count()


@pytest.mark.parametrize(
    "dataset_type",
    [
        ("log_spark"),
        ("log"),
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

    if isinstance(log, pd.DataFrame):
        real_test_size = test.shape[0] / len(log)
        assert train.shape[0] + test.shape[0] == log.shape[0]
    else:
        real_test_size = test.count() / log.count()
        assert train.count() + test.count() == log.count()

    assert np.isclose(real_test_size, 0.6, atol=0.015)
