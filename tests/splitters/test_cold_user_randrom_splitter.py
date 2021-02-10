# pylint: disable-all
from datetime import datetime

import pytest

from replay.constants import LOG_SCHEMA
from replay.splitters import ColdUserRandomSplitter
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
    ).repartition(1)


def test(log):
    cold_user_splitter = ColdUserRandomSplitter(1 / 4)
    cold_user_splitter.seed = 27
    train, test = cold_user_splitter.split(log)
    test_users = test.toPandas().user_id.unique()
    assert len(test_users) == 2  # спарк плевать хотел на weights ¯\_(ツ)_/¯
