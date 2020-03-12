"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту
"""
import os
from datetime import datetime

import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import (IntegerType, StringType, StructField,
                               StructType, TimestampType)
from tests.pyspark_testcase import PySparkTest

import sponge_bob_magic.session_handler
from sponge_bob_magic import utils


class UtilsTestCase(PySparkTest):
    def test_func_get(self):
        vector = np.arange(2)
        self.assertEqual(utils.func_get(vector, 0), 0.0)

    def test_get_feature_cols(self):
        user_features = self.spark.createDataFrame(
            [("1", datetime(2000, 1, 1), 1)],
            schema=StructType([
                StructField("user_id", StringType()),
                StructField("timestamp", TimestampType()),
                StructField("feature1", IntegerType())
            ])
        )
        item_features = self.spark.createDataFrame(
            [("1", datetime(2000, 1, 1), 1), (2, datetime(2000, 1, 1), 0)],
            schema=StructType([
                StructField("item_id", StringType()),
                StructField("timestamp", TimestampType()),
                StructField("feature2", IntegerType())
            ])
        )
        user_feature_cols, item_feature_cols = utils.get_feature_cols(
            user_features, item_features
        )
        self.assertEqual(user_feature_cols, ["feature1"])
        self.assertEqual(item_feature_cols, ["feature2"])

    def test_write_read_dataframe(self):
        dataframe = self.spark.createDataFrame(data=[
            ["user1", "feature1", "2019-01-01"],
            ["user1", "feature2", "2019-01-01"],
            ["user2", "feature1", "2019-01-01"]
        ], schema=["user_id", "features", "timestamp"])
        path = os.path.join(self.spark.conf.get("spark.local.dir"),
                            "tmp_test_write_read_df.parquet")
        test_df = utils.write_read_dataframe(dataframe, path,
                                             to_overwrite_files=True)
        self.assertSparkDataFrameEqual(dataframe, test_df)
        self.assertRaises(
            pyspark.sql.utils.AnalysisException,
            utils.write_read_dataframe,
            dataframe=dataframe, path=path,
            to_overwrite_files=False
        )

    def test_get_spark_session(self):
        spark = sponge_bob_magic.session_handler.get_spark_session(1)
        self.assertIsInstance(spark, SparkSession)
        self.assertEqual(spark.conf.get("spark.driver.memory"), "1g")
