# pylint: disable-all
from datetime import datetime

import numpy
from parameterized import parameterized
from tests.pyspark_testcase import PySparkTest

from replay.constants import LOG_SCHEMA
from replay.splitters.log_splitter import (
    NewUsersSplitter,
    ColdUserRandomSplitter,
    DateSplitter,
    RandomSplitter,
)
from replay.utils import get_distinct_values_in_column


class TestNewUsersSplitter(PySparkTest):
    def test_split(self):
        log = self.spark.createDataFrame(
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

        train, test = NewUsersSplitter(
            test_size=1 / 4, drop_cold_items=False, drop_cold_users=False
        ).split(log=log)
        true_train = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), 1.0],
                ["user4", "item1", datetime(2019, 9, 12), 1.0],
                ["user4", "item2", datetime(2019, 9, 13), 2.0],
                ["user2", "item4", datetime(2019, 9, 14), 3.0],
                ["user2", "item1", datetime(2019, 9, 14), 3.0],
                ["user2", "item2", datetime(2019, 9, 15), 4.0],
            ],
            schema=LOG_SCHEMA,
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user3", "item1", datetime(2019, 9, 16), 5.0],
                ["user3", "item4", datetime(2019, 9, 16), 5.0],
            ],
            schema=LOG_SCHEMA,
        )
        self.assertSparkDataFrameEqual(train, true_train)
        self.assertSparkDataFrameEqual(test, true_test)


class TestColdUserRandomSplitter(PySparkTest):
    def test_split(self):
        log = self.spark.createDataFrame(
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
        true_train = self.spark.createDataFrame(
            data=[
                ["user3", "item1", datetime(2019, 9, 16), 5.0],
                ["user3", "item4", datetime(2019, 9, 16), 5.0],
                ["user4", "item1", datetime(2019, 9, 12), 1.0],
                ["user4", "item2", datetime(2019, 9, 13), 2.0],
            ],
            schema=LOG_SCHEMA,
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), 1.0],
                ["user1", "item3", datetime(2019, 9, 17), 1.0],
                ["user2", "item4", datetime(2019, 9, 14), 3.0],
                ["user2", "item1", datetime(2019, 9, 14), 3.0],
                ["user2", "item2", datetime(2019, 9, 15), 4.0],
            ],
            schema=LOG_SCHEMA,
        )
        cold_user_splitter = ColdUserRandomSplitter(1 / 4)
        cold_user_splitter.seed = 27
        train, test = cold_user_splitter.split(log)
        test.show()
        self.assertSparkDataFrameEqual(test, true_test)
        train.show()
        self.assertSparkDataFrameEqual(train, true_train)
