# pylint: disable-all
from datetime import datetime

import pytest

from replay.constants import LOG_SCHEMA
from replay.splitters import RandomSplitter
from tests.utils import spark


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item4", datetime(2019, 9, 12), 1.0],
            ["user4", "item1", datetime(2019, 9, 12), 1.0],
            ["user4", "item2", datetime(2019, 9, 13), 2.0],
            ["user2", "item4", datetime(2019, 9, 14), 3.0],
            ["user2", "item1", datetime(2019, 9, 14), 3.0],
            ["user2", "item2", datetime(2019, 9, 15), 4.0],
            ["user3", "item1", datetime(2019, 9, 16), 5.0],
            ["user3", "item4", datetime(2019, 9, 16), 5.0],
            ["user1", "item3", datetime(2019, 9, 17), 1.0],
        ],
        schema=LOG_SCHEMA,
    )


test_sizes = [0.0, 1.0, 0.5, 0.22, 0.42, 0.95]


@pytest.mark.parametrize("test_size", test_sizes)
def test_nothing_is_lost(test_size, log):
    splitter = RandomSplitter(
        test_size=test_size,
        drop_cold_items=False,
        drop_cold_users=False,
        seed=1234,
    )
    train, test = splitter.split(log)
    assert train.count() + test.count() == log.count()
