"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

from parameterized import parameterized
from pyspark.sql import DataFrame
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA
from sponge_bob_magic.splitters.base_splitter import SplitterReturnType
from sponge_bob_magic.splitters.user_log_splitter import (
    ByTimeUserLogSplitter, RandomUserLogSplitter, UserLogSplitter)


class TestSplitter(UserLogSplitter):
    def _split_proportion(self, log: DataFrame) -> SplitterReturnType:
        return (log, log, log)

    def _split_quantity(self, log: DataFrame) -> SplitterReturnType:
        return (log, log, log)


class TestUserLogSplitter(PySparkTest):
    def setUp(self):
        data = [
            ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
            ["user2", "item5", datetime(2019, 9, 13), "night", 2.0],
            ["user3", "item7", datetime(2019, 9, 17), "night", 1.0],
            ["user4", "item6", datetime(2019, 9, 17), "night", 1.0],
            ["user5", "item6", datetime(2019, 9, 17), "night", 1.0]
        ]
        self.log = self.spark.createDataFrame(data=data, schema=LOG_SCHEMA)

    @parameterized.expand([(0.5,), (3,)])
    def test_get_test_users(self, fraction):
        test_users = TestSplitter(
            False, False, 1, fraction
        ).get_test_users(self.log)
        self.assertEqual(test_users.count(), 2)
        self.assertSparkDataFrameEqual(
            test_users,
            self.spark.createDataFrame(
                data=[{"user_id": "user4"}, {"user_id": "user5"}]
            )
        )

    @parameterized.expand([(5,), (1.0,)])
    def test_exceptions(self, wrong_fraction):
        with self.assertRaises(ValueError):
            TestSplitter(
                False, False, 1, wrong_fraction
            ).get_test_users(self.log)


class TestRandomUserLogSplitter(PySparkTest):
    def setUp(self):
        self.log = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user1", "item7", datetime(2019, 9, 17), "night", 1.0],
                ["user1", "item6", datetime(2019, 9, 17), "night", 1.0],

                ["user2", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user2", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item6", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user2", "item3", datetime(2019, 9, 15), "night", 4.0],

                ["user3", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user3", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user3", "item6", datetime(2019, 9, 14), "day", 3.0],
                ["user3", "item2", datetime(2019, 9, 14), "day", 3.0],
                ["user3", "item7", datetime(2019, 9, 17), "night", 1.0],

                ["user4", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user4", "item1", datetime(2019, 9, 16), "night", 4.0],
                ["user4", "item4", datetime(2019, 9, 17), "night", 4.0],
                ["user4", "item5", datetime(2019, 9, 18), "night", 4.0],
                ["user4", "item8", datetime(2019, 9, 19), "night", 4.0],
                ["user4", "item4", datetime(2019, 9, 20), "night", 4.0],
                ["user4", "item1", datetime(2019, 9, 21), "night", 4.0],
            ],
            schema=LOG_SCHEMA)

    @parameterized.expand([
        # item_test_size
        (0.1,),
        (0.2,),
        (0.3,),
        (0.4,),
        (0.5,),
        (0.55,),
        (0.6,),
        (0.7,),
        (0.8,),
        (0.9,),
        (1,),
        (2,),
        (3,),
        (4,),
    ])
    def test_split(self, item_test_size):
        train, predict_input, test = (
            RandomUserLogSplitter(
                drop_cold_items=False,
                drop_cold_users=False,
                item_test_size=item_test_size,
                seed=1234)
            .split(log=self.log)
        )

        self.assertSparkDataFrameEqual(train.union(test), self.log)
        self.assertSparkDataFrameEqual(predict_input.union(test), self.log)

        self.assertEqual(test.intersect(train).count(), 0)
        self.assertEqual(test.intersect(predict_input).count(), 0)

        if isinstance(item_test_size, int):
            #  это грубая проверка; чтобы она была верна, необходимо
            # чтобы item_test_size был больше длины лога каждого пользователя
            num_users = self.log.select("user_id").distinct().count()
            self.assertEqual(num_users * item_test_size, test.count())
            self.assertEqual(self.log.count() - num_users * item_test_size,
                             train.count())
            self.assertEqual(self.log.count() - num_users * item_test_size,
                             predict_input.count())

    @parameterized.expand([
        # item_test_size
        (2.0,),
        (2.1,),
        (-1,),
        (-0.01,),
        (-0.01,),
        (-50,),
    ])
    def test_item_test_size_exception(self, item_test_size):
        self.assertRaises(
            ValueError,
            RandomUserLogSplitter(
                self.spark, False, False, item_test_size
            ).split,
            log=self.log
        )


class TestByTimeUserLogSplitter(PySparkTest):
    def setUp(self):
        self.log = self.spark.createDataFrame(
            data=[
                ["1", "1", datetime(2019, 1, 1), "", 1.0],
                ["1", "2", datetime(2019, 1, 2), "", 1.0],
                ["1", "3", datetime(2019, 1, 3), "", 1.0],
                ["1", "4", datetime(2019, 1, 4), "", 1.0],

                ["2", "0", datetime(2020, 2, 5), "", 1.0],
                ["2", "4", datetime(2020, 2, 4), "", 1.0],
                ["2", "3", datetime(2020, 2, 3), "", 1.0],
                ["2", "2", datetime(2020, 2, 2), "", 1.0],
                ["2", "1", datetime(2020, 2, 1), "", 1.0],

                ["3", "1", datetime(1995, 1, 1), "", 1.0],
                ["3", "2", datetime(1995, 1, 2), "", 1.0],
                ["3", "3", datetime(1995, 1, 3), "", 1.0],
            ],
            schema=LOG_SCHEMA)

    def test_split_quantity(self):
        train, predict_input, test = (
            ByTimeUserLogSplitter(
                drop_cold_items=False,
                drop_cold_users=False,
                item_test_size=2)
            .split(log=self.log)
        )

        true_train = self.spark.createDataFrame(
            data=[
                ["1", "1", datetime(2019, 1, 1), "", 1.0],
                ["1", "2", datetime(2019, 1, 2), "", 1.0],

                ["2", "3", datetime(2020, 2, 3), "", 1.0],
                ["2", "2", datetime(2020, 2, 2), "", 1.0],
                ["2", "1", datetime(2020, 2, 1), "", 1.0],

                ["3", "1", datetime(1995, 1, 1), "", 1.0],
            ],
            schema=LOG_SCHEMA)

        true_test = self.spark.createDataFrame(
            data=[
                ["1", "3", datetime(2019, 1, 3), "", 1.0],
                ["1", "4", datetime(2019, 1, 4), "", 1.0],

                ["2", "0", datetime(2020, 2, 5), "", 1.0],
                ["2", "4", datetime(2020, 2, 4), "", 1.0],

                ["3", "2", datetime(1995, 1, 2), "", 1.0],
                ["3", "3", datetime(1995, 1, 3), "", 1.0],
            ],
            schema=LOG_SCHEMA)
        self.assertSparkDataFrameEqual(true_train, train)
        self.assertSparkDataFrameEqual(true_test, test)
        self.assertSparkDataFrameEqual(true_train, predict_input)

    def test_split_proportion(self):
        train, predict_input, test = (
            ByTimeUserLogSplitter(
                drop_cold_items=False,
                drop_cold_users=False,
                item_test_size=0.4)
            .split(log=self.log)
        )

        true_train = self.spark.createDataFrame(
            data=[
                ["1", "1", datetime(2019, 1, 1), "", 1.0],
                ["1", "2", datetime(2019, 1, 2), "", 1.0],
                ["1", "3", datetime(2019, 1, 3), "", 1.0],

                ["2", "3", datetime(2020, 2, 3), "", 1.0],
                ["2", "2", datetime(2020, 2, 2), "", 1.0],
                ["2", "1", datetime(2020, 2, 1), "", 1.0],

                ["3", "1", datetime(1995, 1, 1), "", 1.0],
                ["3", "2", datetime(1995, 1, 2), "", 1.0],
            ],
            schema=LOG_SCHEMA)

        true_test = self.spark.createDataFrame(
            data=[
                ["1", "4", datetime(2019, 1, 4), "", 1.0],

                ["2", "0", datetime(2020, 2, 5), "", 1.0],
                ["2", "4", datetime(2020, 2, 4), "", 1.0],

                ["3", "3", datetime(1995, 1, 3), "", 1.0],
            ],
            schema=LOG_SCHEMA)

        self.assertSparkDataFrameEqual(true_train, train)
        self.assertSparkDataFrameEqual(true_test, test)
        self.assertSparkDataFrameEqual(true_train, predict_input)

    @parameterized.expand([
        # item_test_size
        (2.0,),
        (2.1,),
        (-1,),
        (-0.01,),
        (-0.01,),
        (-50,),
    ])
    def test_item_test_size_exception(self, item_test_size):
        self.assertRaises(
            ValueError,
            ByTimeUserLogSplitter(
                False, False, item_test_size
            ).split,
            log=self.log
        )
