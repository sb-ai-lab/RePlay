import os

import mlflow
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

from experiment_utils import get_model
from replay.experiment import Experiment
from replay.metrics import HitRate, MAP, NDCG
from replay.models import (
    AssociationRulesItemRec,
    ClusterRec,
)
from replay.session_handler import get_spark_session
from replay.utils import (
    JobGroup,
    get_number_of_allocated_executors,
    log_exec_timer,
)
from experiment_utils import get_log_info
from replay.utils import logger


def main(spark: SparkSession, dataset_name: str):
    spark_conf: SparkConf = spark.sparkContext.getConf()

    if spark_conf.get("spark.executor.instances"):
        if get_number_of_allocated_executors(spark) < int(
            spark_conf.get("spark.executor.instances")
        ):
            raise Exception("Not enough executors to run experiment!")

    K = int(os.environ.get("K", 10))
    K_list_metrics = [5, 10]
    SEED = int(os.environ.get("SEED", 1234))
    MLFLOW_TRACKING_URI = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8811"
    )
    MODEL = "ALS"

    if os.environ.get("PARTITION_NUM"):
        partition_num = int(os.environ.get("PARTITION_NUM"))
    else:
        if spark_conf.get("spark.cores.max") is None:
            partition_num = os.cpu_count()
        else:
            partition_num = int(spark_conf.get("spark.cores.max"))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ALS_prediction_on_subset")

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
            "dataset": dataset_name,
            "seed": SEED,
            "K": K,
        }
        mlflow.log_params(spark_configs)

        if dataset_name.startswith("MovieLens"):
            dataset_params = dataset_name.split("__")
            if len(dataset_params) == 1:
                dataset_version = "1m"
            else:
                dataset_version = dataset_params[1]

            with log_exec_timer(
                "Train/test datasets reading to parquet"
            ) as parquets_read_timer:
                train = spark.read.parquet(  # hdfs://node21.bdcl:9000
                    f"/opt/spark_data/replay_datasets/MovieLens/train_{dataset_version}.parquet"
                )
                test = spark.read.parquet(  # hdfs://node21.bdcl:9000
                    f"/opt/spark_data/replay_datasets/MovieLens/test_{dataset_version}.parquet"
                )
            train = train.repartition(partition_num)
            test = test.repartition(partition_num)
            mlflow.log_metric(
                "parquets_read_sec", parquets_read_timer.duration
            )
        elif dataset_name.startswith("MillionSongDataset"):
            # MillionSongDataset__{fraction} pattern
            dataset_params = dataset_name.split("__")
            if len(dataset_params) == 1:
                fraction = "1.0"
            else:
                fraction = dataset_params[1]

            if fraction == "train_100m_users_1k_items":
                train = spark.read.parquet(
                    f"/opt/spark_data/replay_datasets/MillionSongDataset/fraction_{fraction}_train.parquet"
                )
                test = spark.read.parquet(
                    f"/opt/spark_data/replay_datasets/MillionSongDataset/fraction_{fraction}_test.parquet"
                )
                train = train.repartition(partition_num)
                test = test.repartition(partition_num)
            else:
                if partition_num in {6, 12, 24, 48}:
                    with log_exec_timer(
                        "Train/test datasets reading to parquet"
                    ) as parquets_read_timer:
                        train = spark.read.parquet(
                            f"/opt/spark_data/replay_datasets/MillionSongDataset/"
                            f"fraction_{fraction}_train_{partition_num}_partition.parquet"
                        )
                        test = spark.read.parquet(
                            f"/opt/spark_data/replay_datasets/MillionSongDataset/"
                            f"fraction_{fraction}_test_{partition_num}_partition.parquet"
                        )
                else:
                    with log_exec_timer(
                        "Train/test datasets reading to parquet"
                    ) as parquets_read_timer:
                        train = spark.read.parquet(
                            f"/opt/spark_data/replay_datasets/MillionSongDataset/"
                            f"fraction_{fraction}_train_24_partition.parquet"
                        )
                        test = spark.read.parquet(
                            f"/opt/spark_data/replay_datasets/MillionSongDataset/"
                            f"fraction_{fraction}_test_24_partition.parquet"
                        )
                        train = train.repartition(partition_num)
                        test = test.repartition(partition_num)
            mlflow.log_metric(
                "parquets_read_sec", parquets_read_timer.duration
            )

        elif dataset_name == "ml1m":
            with log_exec_timer(
                "Train/test/user_features datasets reading to parquet"
            ) as parquets_read_timer:
                train = spark.read.parquet(
                    "/opt/spark_data/replay_datasets/ml1m_train.parquet"
                )
                test = spark.read.parquet(
                    "/opt/spark_data/replay_datasets/ml1m_test.parquet"
                )
                user_features = spark.read.parquet(
                    "/opt/spark_data/replay_datasets/ml1m_user_features.parquet"
                )
                train = train.repartition(partition_num, "user_idx")
                test = test.repartition(partition_num, "user_idx")
            mlflow.log_metric(
                "parquets_read_sec", parquets_read_timer.duration
            )
        elif dataset_name == "ml1m_1m_users_3_7k_items":
            with log_exec_timer(
                "Train/test/user_features datasets reading to parquet"
            ) as parquets_read_timer:
                train = spark.read.parquet(
                    "hdfs://node21.bdcl:9000/opt/spark_data/replay_datasets/ml1m_1m_users_3_7k_items_train.parquet"
                )
                test = spark.read.parquet(
                    "hdfs://node21.bdcl:9000/opt/spark_data/replay_datasets/ml1m_1m_users_3_7k_items_test.parquet"
                )
                user_features = spark.read.parquet(
                    "hdfs://node21.bdcl:9000/opt/spark_data/replay_datasets/"
                    "ml1m_1m_users_3_7k_items_user_features.parquet"
                )
                print(user_features.printSchema())
                train = train.repartition(partition_num, "user_idx")
                test = test.repartition(partition_num, "user_idx")
            mlflow.log_metric(
                "parquets_read_sec", parquets_read_timer.duration
            )
        elif dataset_name == "ml1m_1m_users_37k_items":
            with log_exec_timer(
                "Train/test/user_features datasets reading to parquet"
            ) as parquets_read_timer:
                train = spark.read.parquet(
                    "/opt/spark_data/replay_datasets/ml1m_1m_users_37k_items_train.parquet"
                )
                test = spark.read.parquet(
                    "/opt/spark_data/replay_datasets/ml1m_1m_users_37k_items_test.parquet"
                )
                user_features = spark.read.parquet(
                    "/opt/spark_data/replay_datasets/ml1m_1m_users_37k_items_user_features.parquet"
                )
                print(user_features.printSchema())
                train = train.repartition(partition_num, "user_idx")
                test = test.repartition(partition_num, "user_idx")
            mlflow.log_metric(
                "parquets_read_sec", parquets_read_timer.duration
            )
        else:
            raise ValueError("Unknown dataset.")

        mlflow.log_param(
            "USE_BUCKETING", os.environ.get("USE_BUCKETING", "False")
        )
        if os.environ.get("USE_BUCKETING", "False") == "True":
            BUCKETING_KEY = "user_idx"

            with log_exec_timer("Train/test caching") as bucketing_timer:
                (
                    train.repartition(partition_num, BUCKETING_KEY)
                    .write.mode("overwrite")
                    .bucketBy(partition_num, BUCKETING_KEY)
                    .sortBy(BUCKETING_KEY)
                    .saveAsTable(
                        f"bucketed_train_{spark.sparkContext.applicationId}",
                        format="parquet",
                        path=f"/spark-warehouse/bucketed_train_{spark.sparkContext.applicationId}",
                    )
                )

                train = spark.table(
                    f"bucketed_train_{spark.sparkContext.applicationId}"
                )

                (
                    test.repartition(partition_num, BUCKETING_KEY)
                    .write.mode("overwrite")
                    .bucketBy(partition_num, BUCKETING_KEY)
                    .sortBy(BUCKETING_KEY)
                    .saveAsTable(
                        f"bucketed_test_{spark.sparkContext.applicationId}",
                        format="parquet",
                        path=f"/spark-warehouse/bucketed_test_{spark.sparkContext.applicationId}",
                    )
                )
                test = spark.table(
                    f"bucketed_test_{spark.sparkContext.applicationId}"
                )

            mlflow.log_metric("bucketing_sec", bucketing_timer.duration)

        with log_exec_timer("Train/test caching") as train_test_cache_timer:
            train = train.cache()
            test = test.cache()
            train.write.mode("overwrite").format("noop").save()
            test.write.mode("overwrite").format("noop").save()
        mlflow.log_metric(
            "train_test_cache_sec", train_test_cache_timer.duration
        )

        mlflow.log_metric("train_num_partitions", train.rdd.getNumPartitions())
        mlflow.log_metric("test_num_partitions", test.rdd.getNumPartitions())

        with log_exec_timer(
            "get_log_info() execution"
        ) as get_log_info_timer:
            train_info = get_log_info(train)
            test_info = get_log_info(test)
            logger.info(
                "train info: total lines: {}, total users: {}, total items: {}".format(
                    *train_info
                )
            )
            logger.info(
                "test info: total lines: {}, total users: {}, total items: {}".format(
                    *test_info
                )
            )
        mlflow.log_metric("get_log_info_sec", get_log_info_timer.duration)
        mlflow.log_param("train_size", train_info[0])
        mlflow.log_param("train.total_users", train_info[1])
        mlflow.log_param("train.total_items", train_info[2])
        mlflow.log_param("test_size", test_info[0])
        mlflow.log_param("test.total_users", test_info[1])
        mlflow.log_param("test.total_items", test_info[2])

        mlflow.log_param("model", MODEL)
        model = get_model(MODEL, SEED, spark.sparkContext.applicationId)

        kwargs = {}
        if isinstance(model, ClusterRec):
            kwargs = {"user_features": user_features}

        with log_exec_timer(f"{MODEL} training") as train_timer, JobGroup(
            "Model training", f"{model.__class__.__name__}.fit()"
        ):
            model.fit(log=train, **kwargs)
        mlflow.log_metric("train_sec", train_timer.duration)

        items_fraction = float(os.environ.get("ITEMS_FRACTION", "0.5"))
        mlflow.log_params({
            "USE_NEW_ALS_METHOD": os.environ.get("USE_NEW_ALS_METHOD", "False"),
            "ITEMS_FRACTION": items_fraction
            }
        )        
        items = model.fit_items.sample(items_fraction, seed=SEED)
        mlflow.log_param(
            "USE_NEW_ALS_METHOD",
            os.environ.get("USE_NEW_ALS_METHOD", "False")
        )

        with log_exec_timer(f"{MODEL} prediction") as infer_timer, JobGroup(
            "Model inference", f"{model.__class__.__name__}.predict()"
        ):
            recs = model.predict(
                k=K,
                users=test.select("user_idx").distinct(),
                items=items,
                log=train,
                filter_seen_items=True,
                **kwargs,
            )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer_sec", infer_timer.duration)

        if os.environ.get("USE_NEW_ALS_METHOD", "False") != "True":
            with log_exec_timer(f"{MODEL} predict filtering") as infer_timer, JobGroup(
                "Predict filtering", "Predict filtering"
            ):
                recs = recs.join(items, on="item_idx")
                recs = recs.cache()
                recs.write.mode("overwrite").format("noop").save()
            mlflow.log_metric("predict_filter_sec", infer_timer.duration)

        if not isinstance(model, AssociationRulesItemRec):
            with log_exec_timer(f"Metrics calculation") as metrics_timer, JobGroup(
                "Metrics calculation", "e.add_result()"
            ):
                e = Experiment(
                    test,
                    {
                        MAP(use_scala_udf=True): K_list_metrics,
                        NDCG(use_scala_udf=True): K_list_metrics,
                        HitRate(use_scala_udf=True): K_list_metrics,
                    },
                )
                e.add_result(MODEL, recs)
            mlflow.log_metric("metrics_sec", metrics_timer.duration)
            for k in K_list_metrics:
                mlflow.log_metric(
                    "NDCG.{}".format(k), e.results.at[MODEL, "NDCG@{}".format(k)]
                )
                mlflow.log_metric(
                    "MAP.{}".format(k), e.results.at[MODEL, "MAP@{}".format(k)]
                )
                mlflow.log_metric(
                    "HitRate.{}".format(k),
                    e.results.at[MODEL, "HitRate@{}".format(k)],
                )


if __name__ == "__main__":
    spark_sess = get_spark_session()
    dataset = os.environ.get(
        "DATASET", "ml1m"
    )
    main(spark=spark_sess, dataset_name=dataset)
    spark_sess.stop()
