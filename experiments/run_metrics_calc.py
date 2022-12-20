import logging.config
import os

import mlflow
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

from replay.experiment import Experiment
from replay.metrics import HitRate, MAP, NDCG
from replay.metrics.mrr import MRR
from replay.metrics.ncis_precision import NCISPrecision
from replay.metrics.precision import Precision
from replay.metrics.recall import Recall
from replay.metrics.rocauc import RocAuc
from replay.metrics.surprisal import Surprisal
from replay.metrics.unexpectedness import Unexpectedness
from replay.session_handler import get_spark_session
from replay.utils import (
    JobGroup,
    getNumberOfAllocatedExecutors,
    log_exec_timer,
)

VERBOSE_LOGGING_FORMAT = (
    "%(asctime)s %(levelname)s %(module)s %(filename)s:%(lineno)d %(message)s"
)
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger("replay")
logger.setLevel(logging.DEBUG)


def main(spark: SparkSession):
    spark_conf: SparkConf = spark.sparkContext.getConf()

    if spark_conf.get("spark.executor.instances"):
        if getNumberOfAllocatedExecutors(spark) < int(
            spark_conf.get("spark.executor.instances")
        ):
            raise Exception("Not enough executors to run experiment!")

    K_list_metrics = [10, 100, 1000]
    MLFLOW_TRACKING_URI = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8811"
    )
    # Predictions from ItemKNN
    MODEL = "ItemKNN"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(
        os.environ.get("EXPERIMENT", "METRICS")
    )

    with mlflow.start_run():

        spark_configs = {
            "spark.driver.cores": spark_conf.get("spark.driver.cores"),
            "spark.driver.memory": spark_conf.get("spark.driver.memory"),
            "spark.memory.fraction": spark_conf.get("spark.memory.fraction"),
            "spark.executor.cores": spark_conf.get("spark.executor.cores"),
            "spark.executor.memory": spark_conf.get("spark.executor.memory"),
            "spark.executor.instances": spark_conf.get(
                "spark.executor.instances"
            ),
            "spark.sql.shuffle.partitions": spark_conf.get(
                "spark.sql.shuffle.partitions"
            ),
            "spark.default.parallelism": spark_conf.get(
                "spark.default.parallelism"
            ),
            "spark.applicationId": spark.sparkContext.applicationId,
        }
        mlflow.log_params(spark_configs)

        dataset = "MillionSongDataset"
        train_path = "hdfs://node21.bdcl:9000/opt/spark_data/replay_datasets/MillionSongDataset/" \
                     "fraction_1.0_train_48_partition.parquet"
        train = spark.read.parquet(train_path)
        test_path = "hdfs://node21.bdcl:9000/opt/spark_data/replay_datasets/MillionSongDataset/" \
                    "fraction_1.0_test_48_partition.parquet"
        test = spark.read.parquet(test_path)
        recs_path = "hdfs://node21.bdcl:9000/tmp/replay/ItemKNN_num_neighbours_10_k_1000_recs_for_metrics_exp.parquet"
        recs = spark.read.parquet(recs_path)

        with log_exec_timer("Input datasets caching"):
            recs = recs.cache()
            test = test.cache()
            train = train.cache()
            recs.write.mode("overwrite").format("noop").save()
            test.write.mode("overwrite").format("noop").save()
            train.write.mode("overwrite").format("noop").save()

        mlflow.log_params({
            "dataset": dataset,
            "test_size": test.count(),
            "test_path": test_path,
            "train_size": train.count(),
            "train_path": train_path,
            "recs_size": recs.count(),
            "recs_path": recs_path,
        })

        if os.environ.get("USE_SCALA_UDFS_METRICS", "False") == "True":
            use_scala_udf = True
        else:
            use_scala_udf = False
        mlflow.log_param("use_scala_udf", use_scala_udf)
            
        with log_exec_timer(f"Metrics calculation") as metrics_timer, JobGroup(
            "Metrics calculation", "e.add_result()"
        ):
            e = Experiment(
                test,
                {
                    Surprisal(train, use_scala_udf=use_scala_udf): K_list_metrics,
                    # WARN: may be need Unexpectedness(recs) init
                    Unexpectedness(train, use_scala_udf=use_scala_udf): K_list_metrics,
                    NCISPrecision(train, use_scala_udf=use_scala_udf): K_list_metrics,
                    MAP(use_scala_udf=use_scala_udf): K_list_metrics,
                    NDCG(use_scala_udf=use_scala_udf): K_list_metrics,
                    HitRate(use_scala_udf=use_scala_udf): K_list_metrics,
                    MRR(use_scala_udf=use_scala_udf): K_list_metrics,
                    Precision(use_scala_udf=use_scala_udf): K_list_metrics,
                    Recall(use_scala_udf=use_scala_udf): K_list_metrics,
                    RocAuc(use_scala_udf=use_scala_udf): K_list_metrics
                },
            )
            e.add_result(MODEL, recs)
        mlflow.log_metric("metrics_sec", metrics_timer.duration)
        metrics = dict()
        for k in K_list_metrics:
            metrics["NDCG.{}".format(k)] = e.results.at[
                MODEL, "NDCG@{}".format(k)
            ]
            metrics["MAP.{}".format(k)] = e.results.at[
                MODEL, "MAP@{}".format(k)
            ]
            metrics["HitRate.{}".format(k)] = e.results.at[
                MODEL, "HitRate@{}".format(k)
            ]
            metrics["MRR.{}".format(k)] = e.results.at[
                MODEL, "MRR@{}".format(k)
            ]
            metrics["Precision.{}".format(k)] = e.results.at[
                MODEL, "Precision@{}".format(k)
            ]
            metrics["Recall.{}".format(k)] = e.results.at[
                MODEL, "Recall@{}".format(k)
            ]
            metrics["RocAuc.{}".format(k)] = e.results.at[
                MODEL, "RocAuc@{}".format(k)
            ]
            metrics["Surprisal.{}".format(k)] = e.results.at[
                MODEL, "Surprisal@{}".format(k)
            ]
            metrics["Unexpectedness.{}".format(k)] = e.results.at[
                MODEL, "Unexpectedness@{}".format(k)
            ]
            metrics["NCISPrecision.{}".format(k)] = e.results.at[
                MODEL, "NCISPrecision@{}".format(k)
            ]
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    spark_sess = get_spark_session()
    main(spark=spark_sess)
    spark_sess.stop()
