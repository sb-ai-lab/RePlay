# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from replay.constants import LOG_SCHEMA
from replay.splitters import NewUsersSplitter
from tests.utils import spark


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            ["user2", "item4", datetime(2019, 9, 14), 3.0],
            ["user2", "item1", datetime(2019, 9, 14), 3.0],
            ["user2", "item2", datetime(2019, 9, 15), 4.0],
            ["user1", "item4", datetime(2019, 9, 12), 1.0],
            ["user4", "item1", datetime(2019, 9, 12), 1.0],
            ["user4", "item2", datetime(2019, 9, 13), 2.0],
            ["user3", "item1", datetime(2019, 9, 16), 5.0],
            ["user3", "item4", datetime(2019, 9, 16), 5.0],
            ["user1", "item3", datetime(2019, 9, 17), 1.0],
        ],
        schema=LOG_SCHEMA,
    )


def test_users_are_cold(log):
    splitter = NewUsersSplitter(
        test_size=0.25, drop_cold_items=False, drop_cold_users=False
    )
    train, test = splitter.split(log)

    train_users = train.toPandas().user_id
    test_users = test.toPandas().user_id

    assert not np.isin(test_users, train_users).any()


@pytest.mark.parametrize("test_size", [-1.0, 2.0])
def test_bad_test_size(log, test_size):
    with pytest.raises(ValueError):
        NewUsersSplitter(test_size)
