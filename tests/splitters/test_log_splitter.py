"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
# pylint: disable-all
from datetime import datetime

import numpy
from parameterized import parameterized
from tests.pyspark_testcase import PySparkTest

from replay.constants import LOG_SCHEMA
from replay.splitters.log_splitter import (
    ColdUserByDateSplitter,
    ColdUserRandomSplitter,
    DateSplitter,
    RandomSplitter,
)
from replay.utils import get_distinct_values_in_column


class TestLogSplitByDateSplitter(PySparkTest):
    def test_split(self):
        log = self.spark.createDataFrame(
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
        train, test = DateSplitter(datetime(2019, 9, 15), False, False).split(
            log
        )

        true_train = self.spark.createDataFrame(
            data=[
                ["user1", "item1", datetime(2019, 9, 12), 1.0],
                ["user1", "item2", datetime(2019, 9, 13), 2.0],
                ["user2", "item1", datetime(2019, 9, 14), 3.0],
            ],
            schema=LOG_SCHEMA,
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item2", datetime(2019, 9, 15), 4.0],
                ["user3", "item1", datetime(2019, 9, 16), 5.0],
                ["user1", "item3", datetime(2019, 9, 17), 1.0],
            ],
            schema=LOG_SCHEMA,
        )
        with self.subTest():
            self.assertSparkDataFrameEqual(true_train, train)

        with self.subTest():
            self.assertSparkDataFrameEqual(true_test, test)

        train, test = DateSplitter(datetime(2019, 9, 15), True, False).split(
            log
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item2", datetime(2019, 9, 15), 4.0],
                ["user3", "item1", datetime(2019, 9, 16), 5.0],
            ],
            schema=LOG_SCHEMA,
        )
        with self.subTest():
            self.assertSparkDataFrameEqual(true_train, train)

        with self.subTest():
            self.assertSparkDataFrameEqual(true_test, test)

        train, test = DateSplitter(datetime(2019, 9, 15), False, True).split(
            log
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item2", datetime(2019, 9, 15), 4.0],
                ["user1", "item3", datetime(2019, 9, 17), 1.0],
            ],
            schema=LOG_SCHEMA,
        )
        with self.subTest():
            self.assertSparkDataFrameEqual(true_train, train)

        with self.subTest():
            self.assertSparkDataFrameEqual(true_test, test)

        train, test = DateSplitter(datetime(2019, 9, 15), True, True).split(
            log
        )
        true_test = self.spark.createDataFrame(
            data=[["user2", "item2", datetime(2019, 9, 15), 4.0]],
            schema=LOG_SCHEMA,
        )
        with self.subTest():
            self.assertSparkDataFrameEqual(true_train, train)

        with self.subTest():
            self.assertSparkDataFrameEqual(true_test, test)


class TestLogSplitRandomlySplitter(PySparkTest):
    @parameterized.expand(
        [
            # test_size, drop_cold_items, drop_cold_users
            (0.0, False, False),
            (1.0, False, False),
            (0.5, True, False),
            (0.6, True, False),
            (0.7, True, False),
            (0.22, False, True),
            (0.35, False, True),
            (0.65, False, True),
            (0.42, True, True),
            (0.75, True, True),
            (0.95, True, True),
        ]
    )
    def test_split(self, test_size, drop_cold_items, drop_cold_users):
        seed = 1234
        log = self.spark.createDataFrame(
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

        train, test = RandomSplitter(
            test_size=test_size,
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            seed=seed,
        ).split(log)

        if not drop_cold_items and not drop_cold_users:
            self.assertSparkDataFrameEqual(log, train.union(test))
            self.assertSparkDataFrameEqual(log, test.union(train))
            self.assertEqual(test.count(), numpy.ceil(log.count() * test_size))

        if drop_cold_items:
            test_items = get_distinct_values_in_column(test, "item_id")
            test_input_items = get_distinct_values_in_column(test, "item_id")
            train_items = get_distinct_values_in_column(test, "item_id")

            self.assertSetEqual(test_items, test_input_items)
            self.assertSetEqual(test_items, train_items)
        if drop_cold_users:
            test_users = get_distinct_values_in_column(test, "user_id")
            test_input_users = get_distinct_values_in_column(test, "user_id")
            train_users = get_distinct_values_in_column(test, "user_id")

            self.assertSetEqual(test_users, test_input_users)
            self.assertSetEqual(test_users, train_users)


class TestColdUserByDateSplitter(PySparkTest):
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

        train, test = ColdUserByDateSplitter(
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
                ["user1", "item4", datetime(2019, 9, 12), 1.0],
                ["user1", "item3", datetime(2019, 9, 17), 1.0],
                ["user3", "item1", datetime(2019, 9, 16), 5.0],
                ["user3", "item4", datetime(2019, 9, 16), 5.0],
                ["user4", "item1", datetime(2019, 9, 12), 1.0],
                ["user4", "item2", datetime(2019, 9, 13), 2.0],
            ],
            schema=LOG_SCHEMA,
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item4", datetime(2019, 9, 14), 3.0],
                ["user2", "item1", datetime(2019, 9, 14), 3.0],
                ["user2", "item2", datetime(2019, 9, 15), 4.0],
            ],
            schema=LOG_SCHEMA,
        )
        cold_user_splitter = ColdUserRandomSplitter(1 / 4)
        cold_user_splitter.seed = 27
        train, test = cold_user_splitter.split(log)
        self.assertSparkDataFrameEqual(test, true_test)
        self.assertSparkDataFrameEqual(train, true_train)
