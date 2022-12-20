import logging.config
import os

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import StructType, IntegerType, StringType, DateType

from replay.data_preparator import Indexer
from replay.session_handler import get_spark_session
from replay.utils import log_exec_timer

VERBOSE_LOGGING_FORMAT = (
    "%(asctime)s %(levelname)s %(module)s %(filename)s:%(lineno)d %(message)s"
)
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger("replay")
logger.setLevel(logging.DEBUG)


def main(spark: SparkSession, dataset_name: str):
    MLFLOW_TRACKING_URI = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8811"
    )

    if dataset_name == "MillionSongDataset10x":
        train70 = (
            spark.read.parquet(
                "file:///opt/spark_data/replay_datasets/MillionSongDataset/posttraining/train70_10x.parquet"
            )
        )
        train80 = (
            spark.read.parquet(
                "file:///opt/spark_data/replay_datasets/MillionSongDataset/posttraining/train80_10x.parquet"
            )
        )
        train_diff80 = (
            spark.read.parquet(
                "file:///opt/spark_data/replay_datasets/MillionSongDataset/posttraining/train_dif80_10x.parquet"
            )
        )
        test = (
            spark.read.parquet(
                "file:///opt/spark_data/replay_datasets/MillionSongDataset/posttraining/train_test_10x.parquet"
            )
        )
    elif dataset_name == "MillionSongDataset":
        schema1 = (
            StructType()
            .add("user_idx", StringType(), True)
            .add("item_idx", StringType(), True)
            .add("relevance", IntegerType(), True)
            .add("timestamp", DateType(), True)
        )
        schema2 = (
            StructType()
            .add("relevance", IntegerType(), True)
            .add("timestamp", DateType(), True)
            .add("user_idx", StringType(), True)
            .add("item_idx", StringType(), True)
        )

        train70 = (
            spark.read.option("header", True)
            .format("csv")
            .schema(schema1)
            .load(
                "file:///opt/spark_data/replay_datasets/MillionSongDataset/posttraining/train70.csv"
            )
        )
        train80 = (
            spark.read.option("header", True)
            .format("csv")
            .schema(schema1)
            .load(
                "file:///opt/spark_data/replay_datasets/MillionSongDataset/posttraining/train80.csv"
            )
        )

        train_diff80 = (
            spark.read.option("header", True)
            .format("csv")
            .schema(schema2)
            .load(
                "file:///opt/spark_data/replay_datasets/MillionSongDataset/posttraining/train_dif80.csv"
            )
        )

        test = (
            spark.read.option("header", True)
            .format("csv")
            .schema(schema1)
            .load(
                "file:///opt/spark_data/replay_datasets/MillionSongDataset/posttraining/test.csv"
            )
        )
    else:
        raise ValueError("Unknown dataset.")

    train70 = train70.select(
        sf.col("user_idx").alias("user_id"),
        sf.col("item_idx").alias("item_id"),
        "relevance",
        sf.unix_timestamp("timestamp").alias("timestamp")
    )
    train80 = train80.select(
        sf.col("user_idx").alias("user_id"),
        sf.col("item_idx").alias("item_id"),
        "relevance",
        sf.unix_timestamp("timestamp").alias("timestamp")
    )

    train_diff80 = train_diff80.select(
        sf.col("user_idx").alias("user_id"),
        sf.col("item_idx").alias("item_id"),
        "relevance",
        sf.unix_timestamp("timestamp").alias("timestamp")
    )

    test = test.select(
        sf.col("user_idx").alias("user_id"),
        sf.col("item_idx").alias("item_id"),
        "relevance",
        sf.unix_timestamp("timestamp").alias("timestamp")
    )

    log = train80.union(test)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(
        os.environ.get("EXPERIMENT", "Dataset_preparation")
    )

    with mlflow.start_run():

        mlflow.log_param(
            "spark.driver.cores",
            spark.sparkContext.getConf().get("spark.driver.cores"),
        )
        mlflow.log_param(
            "spark.driver.memory",
            spark.sparkContext.getConf().get("spark.driver.memory"),
        )
        mlflow.log_param(
            "spark.memory.fraction",
            spark.sparkContext.getConf().get("spark.memory.fraction"),
        )
        mlflow.log_param(
            "spark.executor.cores",
            spark.sparkContext.getConf().get("spark.executor.cores"),
        )
        mlflow.log_param(
            "spark.executor.memory",
            spark.sparkContext.getConf().get("spark.executor.memory"),
        )
        mlflow.log_param(
            "spark.executor.instances",
            spark.sparkContext.getConf().get("spark.executor.instances"),
        )
        mlflow.log_param(
            "spark.sql.shuffle.partitions",
            spark.sparkContext.getConf().get("spark.sql.shuffle.partitions"),
        )
        mlflow.log_param(
            "spark.default.parallelism",
            spark.sparkContext.getConf().get("spark.default.parallelism"),
        )
        mlflow.log_param(
            "spark.applicationId", spark.sparkContext.applicationId
        )

        mlflow.log_param("dataset", "msd_posttraining")

        with log_exec_timer("Indexer training") as indexer_fit_timer:
            indexer = Indexer(user_col="user_id", item_col="item_id")
            indexer.fit(
                users=log.select("user_id"), items=log.select("item_id")
            )
        mlflow.log_metric("indexer_fit_sec", indexer_fit_timer.duration)

        with log_exec_timer("Indexer transform") as indexer_transform_timer:
            train70 = indexer.transform(df=train70)
            train80 = indexer.transform(df=train80)
            train_diff80 = indexer.transform(df=train_diff80)
            test = indexer.transform(df=test)
        mlflow.log_metric(
            "indexer_transform_sec", indexer_transform_timer.duration
        )

        test.printSchema()

        mlflow.log_metric("train_num_partitions", train80.rdd.getNumPartitions())
        mlflow.log_metric("test_num_partitions", test.rdd.getNumPartitions())

        train70.write.mode("overwrite").parquet(
            f"/opt/spark_data/replay_datasets/MillionSongDataset/train70_10x.parquet"
        )
        train80.write.mode("overwrite").parquet(
            f"/opt/spark_data/replay_datasets/MillionSongDataset/train80_10x.parquet"
        )

        train_diff80.write.mode("overwrite").parquet(
            f"/opt/spark_data/replay_datasets/MillionSongDataset/train_diff80_10x.parquet"
        )

        test.write.mode("overwrite").parquet(
            f"/opt/spark_data/replay_datasets/MillionSongDataset/test_10x.parquet"
        )


if __name__ == "__main__":
    spark_sess = get_spark_session()
    dataset = os.getenv("DATASET")
    main(spark=spark_sess, dataset_name=dataset)
    spark_sess.stop()
