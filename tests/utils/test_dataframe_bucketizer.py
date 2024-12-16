import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark.conf import SparkConf

from replay.utils.dataframe_bucketizer import DataframeBucketizer


@pytest.mark.spark
def test_dataframe_bucketizer(spark, log, log2):
    spark_conf: SparkConf = spark.sparkContext.getConf()
    """
    case 1: initialization the bucketizer with table_name
    """
    with DataframeBucketizer(
        bucketing_key="user_idx",
        partition_num=4,
        spark_warehouse_dir=spark_conf.get("spark.sql.warehouse.dir"),
        table_name="bucketed_log",
    ) as bucketizer:
        bucketed_log = bucketizer.transform(log)
        assert spark.catalog._jcatalog.tableExists("bucketed_log")
        assert bucketed_log.count() == log.count()

    """
    case 2: initialization the bucketizer without table_name
    """
    with DataframeBucketizer(
        bucketing_key="user_idx",
        partition_num=2,
        spark_warehouse_dir=spark_conf.get("spark.sql.warehouse.dir"),
    ) as bucketizer:
        with pytest.raises(
            ValueError,
            match="Parameter 'table_name' is not set! " "Please set it via method 'set_table_name'.",
        ):
            bucketed_log = bucketizer.transform(log2)

        bucketizer.set_table_name("bucketed_log2")
        bucketed_log = bucketizer.transform(log2)
        assert spark.catalog._jcatalog.tableExists("bucketed_log2")
        assert bucketed_log.count() == log2.count()
