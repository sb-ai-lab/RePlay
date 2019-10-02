import unittest
import warnings

import pandas as pd
from pyspark.sql import DataFrame, SparkSession


class PySparkTest(unittest.TestCase):
    spark: SparkSession = None
    spark_log_level: str = "ERROR"

    def assertSparkDataFrameEqual(
            self,
            df1: DataFrame,
            df2: DataFrame,
            msg: str or None = None
    ) -> None:
        def _unify_dataframe(df: DataFrame):
            return (df
                    .toPandas()
                    [sorted(df.columns)]
                    .sort_values(by=sorted(df.columns))
                    .reset_index(drop=True))

        try:
            pd.testing.assert_frame_equal(_unify_dataframe(df1),
                                          _unify_dataframe(df2),
                                          check_like=True)
        except AssertionError as e:
            raise self.failureException(msg) from e

    @classmethod
    def create_testing_pyspark_session(cls):
        return (SparkSession
                .builder
                .master("local[1]")
                .config("spark.driver.memory", "512m")
                .config("spark.sql.shuffle.partitions", "1")
                .appName("testing-pyspark")
                .enableHiveSupport()
                .getOrCreate())

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings(action="ignore", category=ResourceWarning)
        cls.spark = cls.create_testing_pyspark_session()
        cls.spark.sparkContext.setLogLevel(cls.spark_log_level)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
