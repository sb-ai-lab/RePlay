from datetime import datetime
from unittest import TestCase

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (FloatType, StringType, StructField, StructType,
                               TimestampType)

from sponge_bob_magic.validation_schemes import ValidationSchemes

LOG_SCHEMA = StructType([
    StructField("user_id", StringType()),
    StructField("item_id", StringType()),
    StructField("timestamp", TimestampType()),
    StructField("context", StringType()),
    StructField("relevance", FloatType())
])


def compare_dataframes(dataframe1: DataFrame, dataframe2: DataFrame) -> bool:
    x = (
        dataframe1.toPandas().sort_values(by=dataframe1.columns)
        .reset_index(drop=True)
    )
    y = (
        dataframe2.toPandas().sort_values(by=dataframe1.columns)
        .reset_index(drop=True)
    )[dataframe1.columns]
    pd.testing.assert_frame_equal(x, y)
    return True


class TestValidationSchemes(TestCase):
    def setUp(self):
        self.spark = (
            SparkSession.builder
            .master("local[1]")
            .config("spark.driver.memory", "512m")
            .getOrCreate()
        )

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
        self.assertTrue(compare_dataframes(true_train, train))
        self.assertTrue(compare_dataframes(train, test_input))
        self.assertTrue(compare_dataframes(true_test, test))
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
        self.assertTrue(compare_dataframes(true_train, train))
        self.assertTrue(compare_dataframes(train, test_input))
        self.assertTrue(compare_dataframes(true_test, test))
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
        self.assertTrue(compare_dataframes(true_train, train))
        self.assertTrue(compare_dataframes(train, test_input))
        self.assertTrue(compare_dataframes(true_test, test))
        train, test_input, test = validation_schemes.log_split_by_date(
            log, datetime(2019, 9, 15), True, True
        )
        true_test = self.spark.createDataFrame(
            data=[
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertTrue(compare_dataframes(true_train, train))
        self.assertTrue(compare_dataframes(train, test_input))
        self.assertTrue(compare_dataframes(true_test, test))
