from datetime import datetime

from sponge_bob_magic.validation_schemes import ValidationSchemes

from constants import LOG_SCHEMA
from pyspark_testcase import PySparkTest


class TestValidationSchemes(PySparkTest):
    def test_log_split_by_date(self):
        validation_schemes = ValidationSchemes(self.spark)
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
        train, test_input, test = validation_schemes.log_split_by_date(
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
        train, test_input, test = validation_schemes.log_split_by_date(
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
        train, test_input, test = validation_schemes.log_split_by_date(
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
        train, test_input, test = validation_schemes.log_split_by_date(
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
