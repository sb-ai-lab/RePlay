"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту
"""

from datetime import datetime

import numpy as np
from pyspark.sql.types import (StructField, StructType, IntegerType,
                               TimestampType, StringType)

from sponge_bob_magic.utils import func_get, get_feature_cols
from tests.pyspark_testcase import PySparkTest


class UtilsTestCase(PySparkTest):
    def test_func_get(self):
        vector = np.arange(2)
        self.assertEqual(func_get(vector, 0), 0.0)

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
        user_feature_cols, item_feature_cols = get_feature_cols(
            user_features, item_features
        )
        self.assertEqual(user_feature_cols, ["feature1"])
        self.assertEqual(item_feature_cols, ["feature2"])
