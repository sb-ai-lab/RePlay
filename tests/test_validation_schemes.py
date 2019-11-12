from datetime import datetime

import numpy as np
from parameterized import parameterized
from pyspark.sql import DataFrame

from pyspark_testcase import PySparkTest
from sponge_bob_magic.constants import LOG_SCHEMA
from sponge_bob_magic.validation_schemes import ValidationSchemes


def get_distinct_values_in_column(df: DataFrame, column: str):
    return set([row[column]
                for row in (df
                            .select(column)
                            .distinct()
                            .collect())
                ])


class TestValidationSchemes(PySparkTest):
    def setUp(self):
        self.splitter = ValidationSchemes(self.spark)
        self.empty_dataframe = self.spark.createDataFrame(
            data=[], schema=LOG_SCHEMA
        )
        self.seed = 1234
        self.mega_log = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user4", "item1", datetime(2019, 9, 12), "day", 1.0],
                ["user4", "item2", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item4", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item1", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user3", "item1", datetime(2019, 9, 16), "day", 5.0],
                ["user3", "item4", datetime(2019, 9, 16), "day", 5.0],
                ["user1", "item3", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )

    def test_log_split_by_date(self):
        log = self.spark.createDataFrame(
            data=[
                ["user1", "item1", datetime(2019, 9, 12), "day", 1.0],
                ["user1", "item2", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item1", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user3", "item1", datetime(2019, 9, 16), "day", 5.0],
                ["user1", "item3", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )
        train, test_input, test = self.splitter.log_split_by_date(
            log, datetime(2019, 9, 15), False, False
        )
        true_train = self.spark.createDataFrame(
            data=[
                ["user1", "item1", datetime(2019, 9, 12), "day", 1.0],
                ["user1", "item2", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item1", datetime(2019, 9, 14), "day", 3.0]
            ],
            schema=LOG_SCHEMA
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user3", "item1", datetime(2019, 9, 16), "day", 5.0],
                ["user1", "item3", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertSparkDataFrameEqual(true_train, train)
        self.assertSparkDataFrameEqual(train, test_input)
        self.assertSparkDataFrameEqual(true_test, test)
        train, test_input, test = self.splitter.log_split_by_date(
            log, datetime(2019, 9, 15), True, False
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user3", "item1", datetime(2019, 9, 16), "day", 5.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertSparkDataFrameEqual(true_train, train)
        self.assertSparkDataFrameEqual(train, test_input)
        self.assertSparkDataFrameEqual(true_test, test)
        train, test_input, test = self.splitter.log_split_by_date(
            log, datetime(2019, 9, 15), False, True
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user1", "item3", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertSparkDataFrameEqual(true_train, train)
        self.assertSparkDataFrameEqual(train, test_input)
        self.assertSparkDataFrameEqual(true_test, test)
        train, test_input, test = self.splitter.log_split_by_date(
            log, datetime(2019, 9, 15), True, True
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertSparkDataFrameEqual(true_train, train)
        self.assertSparkDataFrameEqual(train, test_input)
        self.assertSparkDataFrameEqual(true_test, test)

    @parameterized.expand([
        # test_size, drop_cold_items, drop_cold_users
        (0.0, False, False),
        (0.3, False, False),
        (0.8, False, False),
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
    ])
    def test_log_split_randomly(self, test_size,
                                drop_cold_items, drop_cold_users):
        train, test_input, test = self.splitter.log_split_randomly(
            self.mega_log, test_size=test_size,
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            seed=self.seed
        )
        if not drop_cold_items and not drop_cold_users:
            self.assertSparkDataFrameEqual(self.mega_log, train.union(test))
            self.assertSparkDataFrameEqual(
                self.mega_log, test.union(test_input)
            )
            self.assertEqual(
                test.count(), np.ceil(self.mega_log.count() * test_size)
            )
        if drop_cold_items:
            test_items = get_distinct_values_in_column(test, 'item_id')
            test_input_items = get_distinct_values_in_column(test, 'item_id')
            train_items = get_distinct_values_in_column(test, 'item_id')

            self.assertSetEqual(test_items, test_input_items)
            self.assertSetEqual(test_items, train_items)
        if drop_cold_users:
            test_users = get_distinct_values_in_column(test, 'user_id')
            test_input_users = get_distinct_values_in_column(test, 'user_id')
            train_users = get_distinct_values_in_column(test, 'user_id')

            self.assertSetEqual(test_users, test_input_users)
            self.assertSetEqual(test_users, train_users)

    def test_extract_cold_users(self):
        train, test_input, test = self.splitter.extract_cold_users(
            log=self.mega_log,
            test_size=1 / 4
        )
        self.assertIsNone(test_input)
        true_train = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user4", "item1", datetime(2019, 9, 12), "day", 1.0],
                ["user4", "item2", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item4", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item1", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
            ],
            schema=LOG_SCHEMA
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user3", "item1", datetime(2019, 9, 16), "day", 5.0],
                ["user3", "item4", datetime(2019, 9, 16), "day", 5.0],
            ],
            schema=LOG_SCHEMA
        )
        self.assertSparkDataFrameEqual(train, true_train)
        self.assertSparkDataFrameEqual(test, true_test)

    def test_log_split_randomly_by_user_num(self):
        true_train = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item6", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user3", "item7", datetime(2019, 9, 17), "night", 1.0],
                ["user3", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user3", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user3", "item6", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item3", datetime(2019, 9, 15), "night", 4.0],
                ["user1", "item7", datetime(2019, 9, 17), "night", 1.0],
                ["user2", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user2", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user3", "item2", datetime(2019, 9, 14), "day", 3.0],
                ["user4", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user1", "item6", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )
        train, test_input, test = self.splitter.log_split_randomly_by_user_num(
            log=true_train,
            test_size=3,
            drop_cold_items=False,
            drop_cold_users=False
        )
        train.show()
        test.show()
        self.assertSparkDataFrameEqual(test.union(train), true_train)
        self.assertSparkDataFrameEqual(test.union(test_input), true_train)

        self.assertEqual(test.intersect(train).count(), 0)
        self.assertEqual(test.intersect(test_input).count(), 0)

    def test_log_split_randomly_by_user_frac(self):
        true_train = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item6", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user3", "item7", datetime(2019, 9, 17), "night", 1.0],
                ["user3", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user3", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user3", "item6", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item3", datetime(2019, 9, 15), "night", 4.0],
                ["user1", "item7", datetime(2019, 9, 17), "night", 1.0],
                ["user2", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user2", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user3", "item2", datetime(2019, 9, 14), "day", 3.0],
                ["user4", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user1", "item6", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )
        train, test_input, test = self.splitter.log_split_randomly_by_user_frac(
            log=true_train,
            test_size=0.5,
            drop_cold_items=False,
            drop_cold_users=False
        )
        self.assertSparkDataFrameEqual(train.union(test), true_train)
        self.assertSparkDataFrameEqual(test_input.union(test), true_train)

        self.assertEqual(test.intersect(train).count(), 0)
        self.assertEqual(test.intersect(test_input).count(), 0)
