# pylint: disable-all
import pytest
import numpy as np
import pandas as pd
import pyspark.sql.functions as sf

from replay.splitters import RandomSplitter
from replay.utils import get_spark_session


@pytest.fixture
def log():
    return pd.DataFrame(
        {
            "user_idx": list(range(5000)),
            "item_idx": list(range(5000)),
            "relevance": [1] * 5000,
        }
    )


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
        test_size=[test_size],
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
        RandomSplitter([-1.0, 2.0])


@pytest.mark.parametrize(
    "dataset_type",
    [
        ("log"),
        ("log_spark"),
    ]
)
def test_with_session_ids(dataset_type, request):
    log = request.getfixturevalue(dataset_type)
    if isinstance(log, pd.DataFrame):
        log["session_id"] = [1] * len(log)
        log["timestamp"] = [10] * len(log)
    else:
        log = log.withColumn("session_id", sf.lit(1))
        log = log.withColumn("timestamp", sf.lit(10))

    splitter = RandomSplitter(
        test_size=[0.4],
        drop_cold_items=False,
        drop_cold_users=False,
        session_id_col="session_id",
        seed=SEED,
    )
    train, test = splitter.split(log)

    if isinstance(log, pd.DataFrame):
        real_test_size = test.shape[0] / len(log)
        assert train.shape[0] + test.shape[0] == len(log)
    else:
        real_test_size = test.count() / log.count()
        assert train.count() + test.count() == log.count()
    
    assert np.isclose(real_test_size, 0.4, atol=0.01)
