# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from replay.data import LOG_SCHEMA
from replay.splitters import UserSplitter
from tests.utils import spark


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            [0, 3, datetime(2019, 9, 12), 1.0],
            [1, 4, datetime(2019, 9, 13), 2.0],
            [2, 6, datetime(2019, 9, 17), 1.0],
            [3, 5, datetime(2019, 9, 17), 1.0],
            [4, 5, datetime(2019, 9, 17), 1.0],
            [0, 5, datetime(2019, 9, 12), 1.0],
            [1, 6, datetime(2019, 9, 13), 2.0],
            [2, 7, datetime(2019, 9, 17), 1.0],
            [3, 8, datetime(2019, 9, 17), 1.0],
            [4, 0, datetime(2019, 9, 17), 1.0],
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
    assert np.isin([0, 2, 3], test_users.toPandas().user_idx).all()


@pytest.mark.parametrize("fraction", [5, 1.0])
def test_user_test_size_exception(log, fraction):
    splitter = UserSplitter(
        drop_cold_items=False,
        drop_cold_users=False,
        item_test_size=1,
        user_test_size=fraction,
    )
    with pytest.raises(ValueError):
        splitter._get_test_users(log)


@pytest.fixture
def big_log(spark):
    return spark.createDataFrame(
        data=[
            [0, 3, datetime(2019, 9, 12), 1.0],
            [0, 4, datetime(2019, 9, 13), 2.0],
            [0, 6, datetime(2019, 9, 17), 1.0],
            [0, 5, datetime(2019, 9, 17), 1.0],
            [1, 3, datetime(2019, 9, 12), 1.0],
            [1, 4, datetime(2019, 9, 13), 2.0],
            [1, 5, datetime(2019, 9, 14), 3.0],
            [1, 1, datetime(2019, 9, 15), 4.0],
            [1, 2, datetime(2019, 9, 15), 4.0],
            [2, 3, datetime(2019, 9, 12), 1.0],
            [2, 4, datetime(2019, 9, 13), 2.0],
            [2, 5, datetime(2019, 9, 14), 3.0],
            [2, 1, datetime(2019, 9, 14), 3.0],
            [2, 6, datetime(2019, 9, 17), 1.0],
            [3, 1, datetime(2019, 9, 15), 4.0],
            [3, 0, datetime(2019, 9, 16), 4.0],
            [3, 3, datetime(2019, 9, 17), 4.0],
            [3, 4, datetime(2019, 9, 18), 4.0],
            [3, 7, datetime(2019, 9, 19), 4.0],
            [3, 3, datetime(2019, 9, 20), 4.0],
            [3, 0, datetime(2019, 9, 21), 4.0],
        ],
        schema=LOG_SCHEMA,
    )


test_sizes = np.arange(0.1, 1, 0.25).tolist() + list(range(1, 5))


@pytest.mark.parametrize("item_test_size", test_sizes)
def test_random_split(big_log, item_test_size):
    splitter = UserSplitter(
        drop_cold_items=False,
        drop_cold_users=False,
        item_test_size=item_test_size,
        seed=1234,
    )
    train, test = splitter.split(big_log)
    assert train.count() + test.count() == big_log.count()
    assert test.intersect(train).count() == 0

    if isinstance(item_test_size, int):
        #  it's a rough check. for it to be true, item_test_size must be bigger than log length for every user
        num_users = big_log.select("user_idx").distinct().count()
        assert num_users * item_test_size == test.count()
        assert big_log.count() - num_users * item_test_size == train.count()


@pytest.mark.parametrize("item_test_size", [2.0, -1, -50, 2.1, -0.01])
def test_item_test_size_exception(big_log, item_test_size):
    splitter = UserSplitter(
        drop_cold_items=False,
        drop_cold_users=False,
        item_test_size=item_test_size,
        seed=1234,
    )
    with pytest.raises(ValueError):
        splitter.split(big_log)


@pytest.fixture
def log2(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, datetime(2019, 1, 1), 1.0],
            [0, 1, datetime(2019, 1, 2), 1.0],
            [0, 2, datetime(2019, 1, 3), 1.0],
            [0, 3, datetime(2019, 1, 4), 1.0],
            [1, 4, datetime(2020, 2, 5), 1.0],
            [1, 3, datetime(2020, 2, 4), 1.0],
            [1, 2, datetime(2020, 2, 3), 1.0],
            [1, 1, datetime(2020, 2, 2), 1.0],
            [1, 0, datetime(2020, 2, 1), 1.0],
            [2, 0, datetime(1995, 1, 1), 1.0],
            [2, 1, datetime(1995, 1, 2), 1.0],
            [2, 2, datetime(1995, 1, 3), 1.0],
        ],
        schema=LOG_SCHEMA,
    )


def test_split_quantity(log2):
    splitter = UserSplitter(
        drop_cold_items=False, drop_cold_users=False, item_test_size=2,
    )
    train, test = splitter.split(log2)
    num_items = test.toPandas().user_idx.value_counts()
    assert num_items.nunique() == 1
    assert num_items.unique()[0] == 2


def test_split_proportion(log2):
    splitter = UserSplitter(
        drop_cold_items=False, drop_cold_users=False, item_test_size=0.4,
    )
    train, test = splitter.split(log2)
    num_items = test.toPandas().user_idx.value_counts()
    assert num_items[1] == 2
    assert num_items[0] == 1 and num_items[2] == 1
