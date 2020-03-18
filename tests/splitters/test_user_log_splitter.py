"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

from parameterized import parameterized
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA
from sponge_bob_magic.splitters.user_log_splitter import UserSplitter


class TestUserLogSplitter(PySparkTest):
    def setUp(self):
        data = [
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
        ]
        self.log = self.spark.createDataFrame(data=data, schema=LOG_SCHEMA)

    @parameterized.expand([(3,), (0.6,)])
    def test_get_test_users(self, fraction):
        test_users = UserSplitter(
            drop_cold_items=False, drop_cold_users=False, user_test_size=fraction, seed=1234
        )._get_test_users(self.log)
        self.assertEqual(test_users.count(), 3)
        self.assertSparkDataFrameEqual(
            test_users,
            self.spark.createDataFrame(
                data=[("user2",), ("user4",), ("user5",)]
            ).toDF("user_id")
        )

    @parameterized.expand([(5,), (1.0,)])
    def test_exceptions(self, wrong_fraction):
        with self.assertRaises(ValueError):
            UserSplitter(
                drop_cold_items=False, drop_cold_users=False, item_test_size=1, user_test_size=wrong_fraction
            )._get_test_users(self.log)


class TestRandomUserLogSplitter(PySparkTest):
    def setUp(self):
        self.log = self.spark.createDataFrame(
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
        train, test = (
            UserSplitter(
                drop_cold_items=False,
                drop_cold_users=False,
                item_test_size=item_test_size,
                shuffle=True,
                seed=1234)
            .split(log=self.log)
        )

        self.assertSparkDataFrameEqual(train.union(test), self.log)
        self.assertEqual(test.intersect(train).count(), 0)

        if isinstance(item_test_size, int):
            #  это грубая проверка; чтобы она была верна, необходимо
            # чтобы item_test_size был больше длины лога каждого пользователя
            num_users = self.log.select("user_id").distinct().count()
            self.assertEqual(num_users * item_test_size, test.count())
            self.assertEqual(self.log.count() - num_users * item_test_size,
                             train.count())

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
            UserSplitter(
                drop_cold_items=False,
                drop_cold_users=False,
                item_test_size=item_test_size,
                seed=1234
            ).split,
            log=self.log
        )


class TestByTimeUserLogSplitter(PySparkTest):
    def setUp(self):
        self.log = self.spark.createDataFrame(
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
            schema=LOG_SCHEMA)

    def test_split_quantity(self):
        train, test = (
            UserSplitter(
                drop_cold_items=False,
                drop_cold_users=False,
                item_test_size=2)
            .split(log=self.log)
        )

        true_train = self.spark.createDataFrame(
            data=[
                ["1", "1", datetime(2019, 1, 1), 1.0],
                ["1", "2", datetime(2019, 1, 2), 1.0],

                ["2", "3", datetime(2020, 2, 3), 1.0],
                ["2", "2", datetime(2020, 2, 2), 1.0],
                ["2", "1", datetime(2020, 2, 1), 1.0],

                ["3", "1", datetime(1995, 1, 1), 1.0],
            ],
            schema=LOG_SCHEMA)

        true_test = self.spark.createDataFrame(
            data=[
                ["1", "3", datetime(2019, 1, 3), 1.0],
                ["1", "4", datetime(2019, 1, 4), 1.0],

                ["2", "0", datetime(2020, 2, 5), 1.0],
                ["2", "4", datetime(2020, 2, 4), 1.0],

                ["3", "2", datetime(1995, 1, 2), 1.0],
                ["3", "3", datetime(1995, 1, 3), 1.0],
            ],
            schema=LOG_SCHEMA)
        self.assertSparkDataFrameEqual(true_train, train)
        self.assertSparkDataFrameEqual(true_test, test)

    def test_split_proportion(self):
        train, test = (
            UserSplitter(
                drop_cold_items=False,
                drop_cold_users=False,
                item_test_size=0.4)
            .split(log=self.log)
        )

        true_train = self.spark.createDataFrame(
            data=[
                ["1", "1", datetime(2019, 1, 1), 1.0],
                ["1", "2", datetime(2019, 1, 2), 1.0],
                ["1", "3", datetime(2019, 1, 3), 1.0],

                ["2", "3", datetime(2020, 2, 3), 1.0],
                ["2", "2", datetime(2020, 2, 2), 1.0],
                ["2", "1", datetime(2020, 2, 1), 1.0],

                ["3", "1", datetime(1995, 1, 1), 1.0],
                ["3", "2", datetime(1995, 1, 2), 1.0],
            ],
            schema=LOG_SCHEMA)

        true_test = self.spark.createDataFrame(
            data=[
                ["1", "4", datetime(2019, 1, 4), 1.0],

                ["2", "0", datetime(2020, 2, 5), 1.0],
                ["2", "4", datetime(2020, 2, 4), 1.0],

                ["3", "3", datetime(1995, 1, 3), 1.0],
            ],
            schema=LOG_SCHEMA)

        self.assertSparkDataFrameEqual(true_train, train)
        self.assertSparkDataFrameEqual(true_test, test)

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
            UserSplitter(
                False, False, item_test_size
            ).split,
            log=self.log
        )
