import os

import mlflow
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

from experiment_utils import get_model, get_datasets, make_bucketed_df
from replay.experiment import Experiment
from replay.metrics import HitRate, MAP, NDCG
from replay.model_handler import save, load
from replay.models import (
    AssociationRulesItemRec,
    ClusterRec,
)
from replay.session_handler import get_spark_session
from replay.utils import (
    JobGroup,
    getNumberOfAllocatedExecutors,
    log_exec_timer,
)
from replay.utils import get_log_info2
from replay.utils import logger


def main(spark: SparkSession, dataset_name: str):
    spark_conf: SparkConf = spark.sparkContext.getConf()

    # if enough executors is not allocated in the cluster mode, then we stop the experiment
    if spark_conf.get("spark.executor.instances"):
        if getNumberOfAllocatedExecutors(spark) < int(
            spark_conf.get("spark.executor.instances")
        ):
            raise Exception("Not enough executors to run experiment!")

    k = int(os.environ.get("K", 10))
    k_list_metrics = [5, 10]
    seed = int(os.environ.get("SEED", 1234))
    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8811"
    )
    model_name = os.environ.get("MODEL", "SLIM_NMSLIB_HNSW")
    # LightFM
    # PopRec
    # UserPopRec
    # Word2VecRec Word2VecRec_NMSLIB_HNSW
    # ALS ALS_NMSLIB_HNSW ALS_HNSWLIB
    # SLIM SLIM_NMSLIB_HNSW
    # ItemKNN ItemKNN_NMSLIB_HNSW
    # ClusterRec ClusterRec_HNSWLIB

    if os.environ.get("PARTITION_NUM"):
        partition_num = int(os.environ.get("PARTITION_NUM"))
    else:
        if spark_conf.get("spark.cores.max") is None:
            partition_num = os.cpu_count()
        else:
            partition_num = int(spark_conf.get("spark.cores.max"))

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(os.environ.get("EXPERIMENT", "delete"))

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
            "seed": seed,
            "K": k,
        }
        mlflow.log_params(spark_configs)

        train, test, user_features = get_datasets(
            dataset_name, spark, partition_num
        )

        use_bucketing = os.environ.get("USE_BUCKETING", "False") == "True"
        mlflow.log_param("USE_BUCKETING", use_bucketing)
        if use_bucketing:
            train, train_bucketing_time = make_bucketed_df(
                train,
                spark,
                bucketing_key="user_idx",
                partition_num=partition_num,
                parquet_name=f"bucketed_train_{spark.sparkContext.applicationId.replace('-', '_')}",
            )
            test, test_bucketing_time = make_bucketed_df(
                test,
                spark,
                bucketing_key="user_idx",
                partition_num=partition_num,
                parquet_name=f"bucketed_test_{spark.sparkContext.applicationId.replace('-', '_')}",
            )
            mlflow.log_metric(
                "bucketing_sec", train_bucketing_time + test_bucketing_time
            )

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
            "get_log_info2() execution"
        ) as get_log_info2_timer:
            train_info = get_log_info2(train)
            test_info = get_log_info2(test)
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
        mlflow.log_params(
            {
                "get_log_info_sec": get_log_info2_timer.duration,
                "train.total_users": train_info[1],
                "train.total_items": train_info[2],
                "train_size": train_info[0],
                "test_size": test_info[0],
                "test.total_users": test_info[1],
                "test.total_items": test_info[2],
            }
        )

        mlflow.log_param("model", model_name)
        model = get_model(model_name, seed, spark.sparkContext.applicationId)

        kwargs = {}
        if isinstance(model, ClusterRec):
            kwargs = {"user_features": user_features}

        with log_exec_timer(f"{model_name} training") as train_timer, JobGroup(
            "Model training", f"{model.__class__.__name__}.fit()"
        ):
            model.fit(log=train, **kwargs)
        mlflow.log_metric("train_sec", train_timer.duration)

        with log_exec_timer(f"{model_name} prediction") as infer_timer, JobGroup(
            "Model inference", f"{model.__class__.__name__}.predict()"
        ):
            if isinstance(model, AssociationRulesItemRec):
                recs = model.get_nearest_items(
                    items=test,
                    k=k,
                )
            else:
                recs = model.predict(
                    k=k,
                    users=test.select("user_idx").distinct(),
                    log=train,
                    filter_seen_items=True,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer_sec", infer_timer.duration)

        if not isinstance(model, AssociationRulesItemRec):
            with log_exec_timer(
                f"Metrics calculation"
            ) as metrics_timer, JobGroup(
                "Metrics calculation", "e.add_result()"
            ):
                e = Experiment(
                    test,
                    {
                        MAP(use_scala_udf=True): k_list_metrics,
                        NDCG(use_scala_udf=True): k_list_metrics,
                        HitRate(use_scala_udf=True): k_list_metrics,
                    },
                )
                e.add_result(model_name, recs)
            mlflow.log_metric("metrics_sec", metrics_timer.duration)
            metrics = dict()
            for k in k_list_metrics:
                metrics["NDCG.{}".format(k)] = e.results.at[
                    model_name, "NDCG@{}".format(k)
                ]
                metrics["MAP.{}".format(k)] = e.results.at[
                    model_name, "MAP@{}".format(k)
                ]
                metrics["HitRate.{}".format(k)] = e.results.at[
                    model_name, "HitRate@{}".format(k)
                ]
            mlflow.log_metrics(metrics)

        with log_exec_timer(f"Model saving") as model_save_timer:
            save(
                model,
                path=f"/tmp/replay/{model_name}_{dataset_name}_{spark.sparkContext.applicationId}",  # file://
                overwrite=True,
            )
        mlflow.log_param(
            "model_save_dir",
            f"/tmp/replay/{model_name}_{dataset_name}_{spark.sparkContext.applicationId}",
        )
        mlflow.log_metric("model_save_sec", model_save_timer.duration)

        with log_exec_timer(f"Model loading") as model_load_timer:
            model_loaded = load(
                path=f"/tmp/replay/{model_name}_{dataset_name}_{spark.sparkContext.applicationId}"
            )
        mlflow.log_metric("_loaded_model_sec", model_load_timer.duration)

        with log_exec_timer(
            f"{model_name} prediction from loaded model"
        ) as infer_loaded_timer:
            if isinstance(model_loaded, AssociationRulesItemRec):
                recs = model_loaded.get_nearest_items(
                    items=test,
                    k=k,
                )
            else:
                recs = model_loaded.predict(
                    k=k,
                    users=test.select("user_idx").distinct(),
                    log=train,
                    filter_seen_items=True,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("_loaded_infer_sec", infer_loaded_timer.duration)

        if not isinstance(model, AssociationRulesItemRec):
            with log_exec_timer(
                f"Metrics calculation for loaded model"
            ) as metrics_loaded_timer, JobGroup(
                "Metrics calculation", "e.add_result()"
            ):
                e = Experiment(
                    test,
                    {
                        MAP(use_scala_udf=True): k_list_metrics,
                        NDCG(use_scala_udf=True): k_list_metrics,
                        HitRate(use_scala_udf=True): k_list_metrics,
                    },
                )
                e.add_result(model_name, recs)
            mlflow.log_metric(
                "_loaded_metrics_sec", metrics_loaded_timer.duration
            )
            metrics = dict()
            for k in k_list_metrics:
                metrics["_loaded_NDCG.{}".format(k)] = e.results.at[
                    model_name, "NDCG@{}".format(k)
                ]
                metrics["_loaded_MAP.{}".format(k)] = e.results.at[
                    model_name, "MAP@{}".format(k)
                ]
                metrics["_loaded_HitRate.{}".format(k)] = e.results.at[
                    model_name, "HitRate@{}".format(k)
                ]
            mlflow.log_metrics(metrics)


if __name__ == "__main__":
    spark_sess = get_spark_session()
    dataset = os.environ.get("DATASET", "ml1m")
    main(spark=spark_sess, dataset_name=dataset)
    spark_sess.stop()
