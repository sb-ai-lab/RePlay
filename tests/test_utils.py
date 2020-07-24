"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту
"""
# pylint: disable-all
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    StringType,
    DoubleType,
)

from tests.pyspark_testcase import PySparkTest

import replay.session_handler
from replay import utils


class UtilsTestCase(PySparkTest):
    def test_func_get(self):
        vector = np.arange(2)
        self.assertEqual(utils.func_get(vector, 0), 0.0)

    def test_get_spark_session(self):
        spark = replay.session_handler.get_spark_session(1)
        self.assertIsInstance(spark, SparkSession)
        self.assertEqual(spark.conf.get("spark.driver.memory"), "512m")
        self.assertEqual(spark.conf.get("spark.sql.shuffle.partitions"), "1")

    def setUp(self):
        self.pandas_data_frame = pd.DataFrame(
            [[1, "a", 3.0], [3, "b", 5.0]], columns=["a", "b", "c"]
        )
        self.spark_data_frame = self.spark.createDataFrame(
            [[1, "a", 3.0], [3, "b", 5.0],],
            schema=StructType(
                [
                    StructField("a", LongType()),
                    StructField("b", StringType()),
                    StructField("c", DoubleType()),
                ]
            ),
        )

    def test_pandas_convert(self):
        self.assertSparkDataFrameEqual(
            self.spark_data_frame, utils.convert2spark(self.pandas_data_frame)
        )

    def test_spark_is_unchanged(self):
        spark_data_frame = utils.convert2spark(self.pandas_data_frame)
        self.assertEqual(
            spark_data_frame, utils.convert2spark(spark_data_frame)
        )
