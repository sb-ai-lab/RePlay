from .types import PYSPARK_AVAILABLE, SparkDataFrame

if PYSPARK_AVAILABLE:
    from pyspark.ml import Transformer
    from pyspark.ml.param import Param, Params, TypeConverters
    from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

    from replay.utils.model_handler import get_fs
    from replay.utils.session_handler import State


class DataframeBucketizer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    """
    Buckets the input dataframe, dumps it to spark warehouse directory,
    and returns a bucketed dataframe.
    """

    bucketing_key = Param(
        Params._dummy(),
        "bucketing_key",
        "bucketing key (also used as sort key)",
        typeConverter=TypeConverters.toString,
    )

    partition_num = Param(
        Params._dummy(),
        "partition_num",
        "number of buckets",
        typeConverter=TypeConverters.toInt,
    )

    table_name = Param(
        Params._dummy(),
        "table_name",
        "parquet file name (for storage  in 'spark-warehouse') and spark table name",
        typeConverter=TypeConverters.toString,
    )

    spark_warehouse_dir = Param(
        Params._dummy(),
        "spark_warehouse_dir",
        "spark_warehouse_dir",
        typeConverter=TypeConverters.toString,
    )

    def __init__(
        self,
        bucketing_key: str,
        partition_num: int,
        spark_warehouse_dir: str,
        table_name: str = "",
    ):
        """Makes bucketed dataframe from input dataframe.

        Args:
            bucketing_key: bucketing key (also used as sort key)
            partition_num: number of buckets
            table_name: parquet file name (for storage  in 'spark-warehouse') and spark table name
            spark_warehouse_dir: spark warehouse dir,
                i.e. value of 'spark.sql.warehouse.dir' property
        """
        super().__init__()
        self.set(self.bucketing_key, bucketing_key)
        self.set(self.partition_num, partition_num)
        self.set(self.table_name, table_name)
        self.set(self.spark_warehouse_dir, spark_warehouse_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove_parquet()

    def remove_parquet(self):
        """Removes parquets where bucketed dataset is stored"""
        spark = State().session
        spark_warehouse_dir = self.getOrDefault(self.spark_warehouse_dir)
        table_name = self.getOrDefault(self.table_name)
        fs = get_fs(spark)
        fs_path = spark._jvm.org.apache.hadoop.fs.Path(f"{spark_warehouse_dir}/{table_name}")
        is_exists = fs.exists(fs_path)
        if is_exists:
            fs.delete(fs_path, True)

    def set_table_name(self, table_name: str):
        """Sets table name"""
        self.set(self.table_name, table_name)

    def _transform(self, dataset: SparkDataFrame):
        bucketing_key = self.getOrDefault(self.bucketing_key)
        partition_num = self.getOrDefault(self.partition_num)
        table_name = self.getOrDefault(self.table_name)
        spark_warehouse_dir = self.getOrDefault(self.spark_warehouse_dir)

        if not table_name:
            msg = "Parameter 'table_name' is not set! Please set it via method 'set_table_name'."
            raise ValueError(msg)

        (
            dataset.repartition(partition_num, bucketing_key)
            .write.mode("overwrite")
            .bucketBy(partition_num, bucketing_key)
            .sortBy(bucketing_key)
            .saveAsTable(
                table_name,
                format="parquet",
                path=f"{spark_warehouse_dir}/{table_name}",
            )
        )

        spark = State().session

        return spark.table(table_name)
