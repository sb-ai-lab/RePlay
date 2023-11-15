# pylint: disable-all
import numpy as np
import pandas as pd
import pytest

from replay.splitters import ColdUserRandomSplitter
from replay.utils import get_spark_session


@pytest.fixture
def log():
    return pd.DataFrame(
        {
            "user_id": list(range(5000)),
            "item_id": list(range(5000)),
            "relevance": [1] * 5000,
            "timestamp": [1] * 5000,
        }
    )


@pytest.fixture
def log_spark(log):
    return get_spark_session().createDataFrame(log)


@pytest.mark.parametrize(
    "dataset_type",
    [
        ("log"),
        ("log_spark"),
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
    else:
        test_users = test.toPandas().user_id.unique()
        train_users = train.toPandas().user_id.unique()
        real_ratio = len(test_users) / log.count()

    assert not np.isin(test_users, train_users).any()
    assert np.isclose(
        real_ratio, ratio, atol=0.01
    )  # Spark weights are random ¯\_(ツ)_/¯


def test_invalid_test_size():
    with pytest.raises(ValueError):
        ColdUserRandomSplitter(test_size=1.2, query_column="user_id")
