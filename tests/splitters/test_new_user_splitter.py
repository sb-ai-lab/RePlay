# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from replay.data import LOG_SCHEMA
from replay.splitters import NewUsersSplitter
from tests.utils import spark


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            [1, 3, datetime(2019, 9, 14), 3.0],
            [1, 0, datetime(2019, 9, 14), 3.0],
            [1, 1, datetime(2019, 9, 15), 4.0],
            [0, 3, datetime(2019, 9, 12), 1.0],
            [3, 0, datetime(2019, 9, 12), 1.0],
            [3, 1, datetime(2019, 9, 13), 2.0],
            [2, 0, datetime(2019, 9, 16), 5.0],
            [2, 3, datetime(2019, 9, 16), 5.0],
            [0, 2, datetime(2019, 9, 17), 1.0],
        ],
        schema=LOG_SCHEMA,
    )


def test_users_are_cold(log):
    splitter = NewUsersSplitter(test_size=0.25, drop_cold_items=False)
    train, test = splitter.split(log)

    train_users = train.toPandas().user_idx
    test_users = test.toPandas().user_idx

    assert not np.isin(test_users, train_users).any()


@pytest.mark.parametrize("test_size", [-1.0, 2.0])
def test_bad_test_size(log, test_size):
    with pytest.raises(ValueError):
        NewUsersSplitter(test_size)
