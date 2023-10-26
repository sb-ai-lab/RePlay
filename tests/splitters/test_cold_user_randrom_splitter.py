# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np
import pandas as pd

from replay.splitters import ColdUserRandomSplitter
from replay.utils import get_spark_session


@pytest.fixture
def log():
    return pd.DataFrame(
        {
            "user_idx": list(range(5000)),
            "item_idx": list(range(5000)),
            "session_id": [1] * 5000,
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
    cold_user_splitter = ColdUserRandomSplitter(ratio, session_id_col="session_id")
    cold_user_splitter.seed = 27
    train, test = cold_user_splitter.split(log)

    if isinstance(log, pd.DataFrame):
        test_users = test.user_idx.unique()
        train_users = train.user_idx.unique()
        real_ratio = len(test_users) / len(log)
    else:
        test_users = test.toPandas().user_idx.unique()
        train_users = train.toPandas().user_idx.unique()
        real_ratio = len(test_users) / log.count()

    assert not np.isin(test_users, train_users).any()
    assert np.isclose(
        real_ratio, ratio, atol=0.01
    )  # Spark weights are random ¯\_(ツ)_/¯


def test_invalid_test_size():
    with pytest.raises(ValueError):
        ColdUserRandomSplitter(test_size=1.2)
