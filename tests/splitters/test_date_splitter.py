# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from replay.data import LOG_SCHEMA
from replay.splitters import DateSplitter
from tests.utils import spark


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, datetime(2019, 9, 12), 1.0],
            [0, 1, datetime(2019, 9, 13), 2.0],
            [1, 0, datetime(2019, 9, 14), 3.0],
            [1, 1, datetime(2019, 9, 15), 4.0],
            [2, 0, datetime(2019, 9, 16), 5.0],
            [0, 2, datetime(2019, 9, 17), 1.0],
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


def test_string(log, split_date):
    splitter = DateSplitter(
        split_date, drop_cold_items=False, drop_cold_users=False
    )
    train_by_date, test_by_date = splitter.split(log)

    str_date = split_date.strftime("%Y-%m-%d")
    splitter = DateSplitter(
        str_date, drop_cold_items=False, drop_cold_users=False
    )
    train_by_str, test_by_str = splitter.split(log)

    int_date = int(split_date.timestamp())
    splitter = DateSplitter(
        int_date, drop_cold_items=False, drop_cold_users=False
    )
    train_by_int, test_by_int = splitter.split(log)

    assert train_by_date.count() == train_by_str.count()
    assert test_by_date.count() == test_by_str.count()

    assert train_by_date.count() == train_by_int.count()
    assert test_by_date.count() == test_by_int.count()


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

    train_items = train.toPandas().item_idx
    test_items = test.toPandas().item_idx

    assert np.isin(test_items, train_items).all()


def test_drop_cold_users(log, split_date):
    splitter = DateSplitter(
        split_date, drop_cold_items=False, drop_cold_users=True
    )
    train, test = splitter.split(log)

    train_users = train.toPandas().user_idx
    test_users = test.toPandas().user_idx

    assert np.isin(test_users, train_users).all()
