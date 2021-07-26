# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from replay.constants import LOG_SCHEMA
from replay.splitters import DateSplitter
from tests.utils import spark


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", datetime(2019, 9, 12), 1.0],
            ["user1", "item2", datetime(2019, 9, 13), 2.0],
            ["user2", "item1", datetime(2019, 9, 14), 3.0],
            ["user2", "item2", datetime(2019, 9, 15), 4.0],
            ["user3", "item1", datetime(2019, 9, 16), 5.0],
            ["user1", "item3", datetime(2019, 9, 17), 1.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def split_date():
    return datetime(2019, 9, 15)


def test_split(log, split_date):
    splitter = DateSplitter(
        split_date, drop_cold_items=False, drop_cold_users=False
    )
    train, test = splitter.split(log)

    train_max_date = train.toPandas().timestamp.max()
    test_min_date = test.toPandas().timestamp.min()

    assert train_max_date < split_date
    assert test_min_date >= split_date


def test_proportion(log):
    test_size = 0.15
    splitter = DateSplitter(test_size)
    train, test = splitter.split(log)

    train_max_date = train.toPandas().timestamp.max()
    test_min_date = test.toPandas().timestamp.min()
    split_date = datetime(2019, 9, 17)

    assert train_max_date < split_date
    assert test_min_date >= split_date
    assert np.isclose(test.count() / log.count(), test_size, atol=0.1)


def test_drop_cold_items(log, split_date):
    splitter = DateSplitter(
        split_date, drop_cold_items=True, drop_cold_users=False
    )
    train, test = splitter.split(log)

    train_items = train.toPandas().item_id
    test_items = test.toPandas().item_id

    assert np.isin(test_items, train_items).all()


def test_drop_cold_users(log, split_date):
    splitter = DateSplitter(
        split_date, drop_cold_items=False, drop_cold_users=True
    )
    train, test = splitter.split(log)

    train_users = train.toPandas().user_id
    test_users = test.toPandas().user_id

    assert np.isin(test_users, train_users).all()
