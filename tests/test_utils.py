"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту
"""
# pylint: disable-all
import numpy as np
from pyspark.sql import SparkSession
from tests.pyspark_testcase import PySparkTest

import sponge_bob_magic.session_handler
from sponge_bob_magic import utils


class UtilsTestCase(PySparkTest):
    def test_func_get(self):
        vector = np.arange(2)
        self.assertEqual(utils.func_get(vector, 0), 0.0)

    def test_get_spark_session(self):
        spark = sponge_bob_magic.session_handler.get_spark_session(1)
        self.assertIsInstance(spark, SparkSession)
        self.assertEqual(spark.conf.get("spark.driver.memory"), "1g")
