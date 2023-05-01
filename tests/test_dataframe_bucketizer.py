# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
from tests.utils import (
    log,
    spark,
)

from pyspark.conf import SparkConf

from replay.dataframe_bucketizer import DataframeBucketizer


def test_dataframe_bucketizer(spark, log):
    spark_conf: SparkConf = spark.sparkContext.getConf()
    with DataframeBucketizer(
        bucketing_key="user_idx",
        partition_num=4,
        spark_warehouse_dir=spark_conf.get("spark.sql.warehouse.dir"),
        table_name="bucketed_log",
    ) as bucketizer:
        bucketed_log = bucketizer.transform(log)
        assert spark.catalog._jcatalog.tableExists("bucketed_log")
        assert bucketed_log.count() == log.count()
