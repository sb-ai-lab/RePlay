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
            ["user1", "item4", datetime(2019, 9, 12), 1.0],
            ["user1", "item5", datetime(2019, 9, 13), 2.0],
            ["user1", "item7", datetime(2019, 9, 17), 1.0],
            ["user1", "item6", datetime(2019, 9, 17), 1.0],
            ["user2", "item4", datetime(2019, 9, 12), 1.0],
            ["user2", "item5", datetime(2019, 9, 13), 2.0],
            ["user2", "item6", datetime(2019, 9, 14), 3.0],
            ["user2", "item2", datetime(2019, 9, 15), 4.0],
            ["user2", "item3", datetime(2019, 9, 15), 4.0],
            ["user3", "item4", datetime(2019, 9, 12), 1.0],
            ["user3", "item5", datetime(2019, 9, 13), 2.0],
            ["user3", "item6", datetime(2019, 9, 14), 3.0],
            ["user3", "item2", datetime(2019, 9, 14), 3.0],
            ["user3", "item7", datetime(2019, 9, 17), 1.0],
            ["user4", "item2", datetime(2019, 9, 15), 4.0],
            ["user4", "item1", datetime(2019, 9, 16), 4.0],
            ["user4", "item4", datetime(2019, 9, 17), 4.0],
            ["user4", "item5", datetime(2019, 9, 18), 4.0],
            ["user4", "item8", datetime(2019, 9, 19), 4.0],
            ["user4", "item4", datetime(2019, 9, 20), 4.0],
            ["user4", "item1", datetime(2019, 9, 21), 4.0],
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
        #  это грубая проверка; чтобы она была верна, необходимо
        # чтобы item_test_size был больше длины лога каждого пользователя
        num_users = big_log.select("user_id").distinct().count()
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
            ["1", "1", datetime(2019, 1, 1), 1.0],
            ["1", "2", datetime(2019, 1, 2), 1.0],
            ["1", "3", datetime(2019, 1, 3), 1.0],
            ["1", "4", datetime(2019, 1, 4), 1.0],
            ["2", "0", datetime(2020, 2, 5), 1.0],
            ["2", "4", datetime(2020, 2, 4), 1.0],
            ["2", "3", datetime(2020, 2, 3), 1.0],
            ["2", "2", datetime(2020, 2, 2), 1.0],
            ["2", "1", datetime(2020, 2, 1), 1.0],
            ["3", "1", datetime(1995, 1, 1), 1.0],
            ["3", "2", datetime(1995, 1, 2), 1.0],
            ["3", "3", datetime(1995, 1, 3), 1.0],
        ],
        schema=LOG_SCHEMA,
    )


def test_split_quantity(log2):
    splitter = UserSplitter(
        drop_cold_items=False, drop_cold_users=False, item_test_size=2,
    )
    train, test = splitter.split(log2)
    num_items = test.toPandas().user_id.value_counts()
    assert num_items.nunique() == 1
    assert num_items.unique()[0] == 2


def test_split_proportion(log2):
    splitter = UserSplitter(
        drop_cold_items=False, drop_cold_users=False, item_test_size=0.4,
    )
    train, test = splitter.split(log2)
    num_items = test.toPandas().user_id.value_counts()
    assert num_items["2"] == 2
    assert num_items["1"] == 1 and num_items["3"] == 1
