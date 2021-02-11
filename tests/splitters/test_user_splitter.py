# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from replay.constants import LOG_SCHEMA
from replay.splitters import UserSplitter
from tests.utils import spark


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item4", datetime(2019, 9, 12), 1.0],
            ["user2", "item5", datetime(2019, 9, 13), 2.0],
            ["user3", "item7", datetime(2019, 9, 17), 1.0],
            ["user4", "item6", datetime(2019, 9, 17), 1.0],
            ["user5", "item6", datetime(2019, 9, 17), 1.0],
            ["user1", "item6", datetime(2019, 9, 12), 1.0],
            ["user2", "item7", datetime(2019, 9, 13), 2.0],
            ["user3", "item8", datetime(2019, 9, 17), 1.0],
            ["user4", "item9", datetime(2019, 9, 17), 1.0],
            ["user5", "item1", datetime(2019, 9, 17), 1.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.mark.parametrize("fraction", [3, 0.6])
def test_get_test_users(log, fraction):
    splitter = UserSplitter(
        drop_cold_items=False,
        drop_cold_users=False,
        user_test_size=fraction,
        seed=1234,
    )
    test_users = splitter._get_test_users(log)
    assert test_users.count() == 3
    assert np.isin(
        ["user1", "user3", "user4"], test_users.toPandas().user_id
    ).all()


@pytest.mark.parametrize("fraction", [5, 1.0])
def test_exceptions(log, fraction):
    splitter = UserSplitter(
        drop_cold_items=False,
        drop_cold_users=False,
        item_test_size=1,
        user_test_size=fraction,
    )
    with pytest.raises(ValueError):
        splitter._get_test_users(log)
