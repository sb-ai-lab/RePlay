import unittest

import pandas as pd
from pyspark.sql import SparkSession, DataFrame


class PySparkTest(unittest.TestCase):
    spark: SparkSession = None
    spark_log_level: str = 'ERROR'

    def assertSparkDataFrameEqual(self,
                                  df1: DataFrame,
                                  df2: DataFrame,
                                  msg: str or None = None) \
            -> None:
        df1 = df1.toPandas().sort_values(by=df1.columns).reset_index(drop=True)
        df2 = df2.toPandas().sort_values(by=df2.columns).reset_index(drop=True)

        try:
            pd.testing.assert_frame_equal(df1, df2, check_like=True)
        except AssertionError as e:
            raise self.failureException(msg) from e

    @classmethod
    def create_testing_pyspark_session(cls):
        return (SparkSession
                .builder
                .master('local[1]')
                .config('spark.driver.memory', '512m')
                .appName('testing-pyspark')
                .enableHiveSupport()
                .getOrCreate())

    @classmethod
    def setUpClass(cls):
        cls.spark = cls.create_testing_pyspark_session()
        cls.spark.sparkContext.setLogLevel(cls.spark_log_level)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
