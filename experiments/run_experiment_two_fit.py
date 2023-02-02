import logging.config
import os

import mlflow
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, IntegerType

from experiment_utils import (
    get_model,
    get_partition_num,
    get_spark_configs_as_dict,
    get_log_info,
)
from replay.dataframe_bucketizer import DataframeBucketizer
from replay.experiment import Experiment
from replay.metrics import HitRate, MAP, NDCG
from replay.model_handler import save, load
from replay.models import AssociationRulesItemRec, UserPopRec
from replay.session_handler import get_spark_session
from replay.utils import (
    JobGroup,
    get_number_of_allocated_executors,
    log_exec_timer,
)
from replay.utils import logger

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

spark_logger = logging.getLogger("py4j")
spark_logger.setLevel(logging.WARN)

formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)
hdlr = logging.StreamHandler()
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


def main(spark: SparkSession, dataset_name: str):
    spark_conf: SparkConf = spark.sparkContext.getConf()

    if spark_conf.get("spark.executor.instances"):
        if get_number_of_allocated_executors(spark) < int(
            spark_conf.get("spark.executor.instances")
        ):
            raise Exception("Not enough executors to run experiment!")

    k = int(os.environ.get("K", 10))
    k_list_metrics = list(map(int, os.environ["K_LIST_METRICS"].split(",")))
    seed = int(os.environ.get("SEED", 1234))
    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8822"
    )
    model_name = os.environ["MODEL"]

    # model_name = os.environ.get("MODEL", "ALS_NMSLIB_HNSW")
    # PopRec
    # Word2VecRec Word2VecRec_NMSLIB_HNSW
    # ALS ALS_NMSLIB_HNSW
    # SLIM SLIM_NMSLIB_HNSW
    # ItemKNN ItemKNN_NMSLIB_HNSW
    # ClusterRec
    # UCB

    partition_num = get_partition_num(spark_conf)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(os.environ.get("EXPERIMENT", "delete"))

    with mlflow.start_run():
        params = get_spark_configs_as_dict(spark_conf)
        params.update(
            {
                "spark.applicationId": spark.sparkContext.applicationId,
                "dataset": dataset_name,
                "seed": seed,
                "K": k,
            }
        )
        mlflow.log_params(params)

        if dataset_name == "ml1m":
            schema = (
                StructType()
                .add("relevance", IntegerType(), True)
                .add("timestamp", IntegerType(), True)
                .add("user_idx", IntegerType(), True)
                .add("item_idx", IntegerType(), True)
            )

            train70 = (
                spark.read.option("header", True)
                .format("csv")
                .schema(schema)
                .load(
                    "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/train70_ml1m.csv"
                )
            )
            train80 = (
                spark.read.option("header", True)
                .format("csv")
                .schema(schema)
                .load(
                    "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/train80_ml1m.csv"
                )
            )
            train_diff80 = (
                spark.read.option("header", True)
                .format("csv")
                .schema(schema)
                .load(
                    "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/train_dif80_ml1m.csv"
                )
            )
            test = (
                spark.read.option("header", True)
                .format("csv")
                .schema(schema)
                .load(
                    "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/test_ml1m.csv"
                )
            )
        elif dataset_name == "MillionSongDataset":
            train70 = spark.read.parquet(
                "/opt/spark_data/replay_datasets/MillionSongDataset/train70.parquet"
            )
            train80 = spark.read.parquet(
                "/opt/spark_data/replay_datasets/MillionSongDataset/train80.parquet"
            )
            train_diff80 = spark.read.parquet(
                "/opt/spark_data/replay_datasets/MillionSongDataset/train_diff80.parquet"
            )
            test = spark.read.parquet(
                "/opt/spark_data/replay_datasets/MillionSongDataset/test.parquet"
            )
        elif dataset_name == "MillionSongDataset10x":
            train70 = spark.read.parquet(
                "/opt/spark_data/replay_datasets/MillionSongDataset/train70_10x.parquet"
            )
            train80 = spark.read.parquet(
                "/opt/spark_data/replay_datasets/MillionSongDataset/train80_10x.parquet"
            )
            train_diff80 = spark.read.parquet(
                "/opt/spark_data/replay_datasets/MillionSongDataset/train_diff80_10x.parquet"
            )
            test = spark.read.parquet(
                "/opt/spark_data/replay_datasets/MillionSongDataset/test_10x.parquet"
            )
        else:
            ValueError("Unknown dataset.")

        kwargs = {}
        if model_name == "ClusterRec":
            user_features = spark.read.parquet(
                "file:///opt/spark_data/replay_datasets/MovieLens/train80_ml1m_user_features.parquet"
            )
            kwargs = {"user_features": user_features}

        use_bucketing = os.environ.get("USE_BUCKETING", "False") == "True"
        mlflow.log_param("USE_BUCKETING", use_bucketing)
        if use_bucketing:
            bucketizer = DataframeBucketizer(
                bucketing_key="user_idx",
                partition_num=partition_num,
                spark_warehouse_dir=spark_conf.get("spark.sql.warehouse.dir"),
            )

            with log_exec_timer("dataframe bucketing") as bucketing_timer:
                bucketizer.set_table_name(
                    f"bucketed_train70_{spark.sparkContext.applicationId.replace('-', '_')}"
                )
                train70 = bucketizer.transform(train70)

                bucketizer.set_table_name(
                    f"bucketed_test_{spark.sparkContext.applicationId.replace('-', '_')}"
                )
                test = bucketizer.transform(test)
            mlflow.log_metric("bucketing_sec", bucketing_timer.duration)

        with log_exec_timer("Train/test caching") as train_test_cache_timer:
            train70 = train70.cache()
            train80 = train80.cache()
            train_diff80 = train_diff80.cache()
            test = test.cache()
            train70.write.mode("overwrite").format("noop").save()
            train80.write.mode("overwrite").format("noop").save()
            train_diff80.write.mode("overwrite").format("noop").save()
            test.write.mode("overwrite").format("noop").save()
        mlflow.log_metric(
            "train_test_cache_sec", train_test_cache_timer.duration
        )

        mlflow.log_metric(
            "train_num_partitions", train70.rdd.getNumPartitions()
        )
        mlflow.log_metric("test_num_partitions", test.rdd.getNumPartitions())

        with log_exec_timer("get_log_info() execution") as get_log_info_timer:
            (
                train_rows_count,
                train_users_count,
                train_items_count,
            ) = get_log_info(train80)
            test_rows_count, test_users_count, test_items_count = get_log_info(
                test
            )
            logger.info(
                f"train info: total lines: {train_rows_count}, "
                f"total users: {train_users_count}, "
                f"total items: {train_items_count}"
            )
            logger.info(
                f"test info: total lines: {test_rows_count}, "
                f"total users: {test_users_count}, "
                f"total items: {test_items_count}"
            )
        mlflow.log_params(
            {
                "get_log_info_sec": get_log_info_timer.duration,
                "train.total_users": train_users_count,
                "train.total_items": train_items_count,
                "train_size": train_rows_count,
                "test_size": test_rows_count,
                "test.total_users": test_users_count,
                "test.total_items": test_items_count,
            }
        )

        mlflow.log_param("model", model_name)
        model = get_model(model_name, seed, spark.sparkContext.applicationId)

        filter_seen_items = True
        if isinstance(model, UserPopRec):
            filter_seen_items = False

        print("\ndataset: train70")
        with log_exec_timer(
            f"{model_name} training on train70 dataset"
        ) as train_timer, JobGroup(
            "Model training", f"{model.__class__.__name__}.fit()"
        ):
            model.fit(log=train70, **kwargs)
        mlflow.log_metric("train70_sec", train_timer.duration)

        with log_exec_timer(
            f"{model_name} prediction"
        ) as infer_timer, JobGroup(
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
                    log=train70,
                    filter_seen_items=filter_seen_items,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer70_sec", infer_timer.duration)

        if os.environ.get("USE_SCALA_UDFS_METRICS", "False") == "True":
            use_scala_udf = True
        else:
            use_scala_udf = False
        mlflow.log_param("use_scala_udf", use_scala_udf)

        if not isinstance(model, AssociationRulesItemRec):
            with log_exec_timer(
                f"Metrics calculation"
            ) as metrics_timer, JobGroup(
                "Metrics calculation", "e.add_result()"
            ):
                e = Experiment(
                    test,
                    {
                        MAP(use_scala_udf=use_scala_udf): k_list_metrics,
                        NDCG(use_scala_udf=use_scala_udf): k_list_metrics,
                        HitRate(use_scala_udf=use_scala_udf): k_list_metrics,
                    },
                )
                e.add_result(model_name, recs)
            mlflow.log_metric("metrics70_sec", metrics_timer.duration)
            metrics = dict()
            for k in k_list_metrics:
                metrics["NDCG.{}_70".format(k)] = e.results.at[
                    model_name, "NDCG@{}".format(k)
                ]
                metrics["MAP.{}_70".format(k)] = e.results.at[
                    model_name, "MAP@{}".format(k)
                ]
                metrics["HitRate.{}_70".format(k)] = e.results.at[
                    model_name, "HitRate@{}".format(k)
                ]
            mlflow.log_metrics(metrics)

        with log_exec_timer(f"Model saving") as model_save_timer:
            save(
                model,
                path=f"/tmp/replay/{model_name}_{dataset_name}_{spark.sparkContext.applicationId}",  # file://
                overwrite=True,
            )
        del model
        mlflow.log_param(
            "model_save_dir",
            f"/tmp/replay/{model_name}_{dataset_name}_{spark.sparkContext.applicationId}",
        )
        mlflow.log_metric("model_save_sec", model_save_timer.duration)

        with log_exec_timer(f"Model loading") as model_load_timer:
            loaded_model = load(
                path=f"/tmp/replay/{model_name}_{dataset_name}_{spark.sparkContext.applicationId}"
            )
        mlflow.log_metric("model_load_sec", model_load_timer.duration)

        print("\ndataset: train_diff")
        with log_exec_timer(
            f"{model_name} training (additional)"
        ) as train_timer, JobGroup(
            "Model training (additional)", f"{type(loaded_model).__name__}.fit()"
        ):
            loaded_model.fit_partial(log=train_diff80, previous_log=train70)
        mlflow.log_metric("train_diff_sec", train_timer.duration)

        with log_exec_timer(
            f"{model_name} prediction"
        ) as infer_timer, JobGroup(
            "Model inference", f"{loaded_model.__class__.__name__}.predict()"
        ):
            if isinstance(loaded_model, AssociationRulesItemRec):
                recs = loaded_model.get_nearest_items(
                    items=test,
                    k=k,
                )
            else:
                recs = loaded_model.predict(
                    k=k,
                    users=test.select("user_idx").distinct(),
                    log=train80,
                    filter_seen_items=filter_seen_items,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer_diff_sec", infer_timer.duration)

        if not isinstance(loaded_model, AssociationRulesItemRec):
            with log_exec_timer(
                f"Metrics calculation"
            ) as metrics_timer, JobGroup(
                "Metrics calculation", "e.add_result()"
            ):
                e = Experiment(
                    test,
                    {
                        MAP(use_scala_udf=use_scala_udf): k_list_metrics,
                        NDCG(use_scala_udf=use_scala_udf): k_list_metrics,
                        HitRate(use_scala_udf=use_scala_udf): k_list_metrics,
                    },
                )
                e.add_result(model_name, recs)
            mlflow.log_metric("metrics_diff_sec", metrics_timer.duration)
            metrics = dict()
            for k in k_list_metrics:
                metrics["NDCG.{}_diff".format(k)] = e.results.at[
                    model_name, "NDCG@{}".format(k)
                ]
                metrics["MAP.{}_diff".format(k)] = e.results.at[
                    model_name, "MAP@{}".format(k)
                ]
                metrics["HitRate.{}_diff".format(k)] = e.results.at[
                    model_name, "HitRate@{}".format(k)
                ]
            mlflow.log_metrics(metrics)

        del loaded_model

        # unpersist all caches
        for (_id, rdd) in spark.sparkContext._jsc.getPersistentRDDs().items():
            rdd.unpersist()
            print("Unpersisted {} rdd".format(_id))
        test = test.cache()
        train80 = train80.cache()
        train80.write.mode("overwrite").format("noop").save()
        test.write.mode("overwrite").format("noop").save()

        print("\ndataset: train80")
        # del model
        model2 = get_model(model_name, seed, spark.sparkContext.applicationId)
        with log_exec_timer(f"{model_name} training") as train_timer, JobGroup(
            "Model training", f"{model2.__class__.__name__}.fit()"
        ):
            model2.fit(log=train80, **kwargs)
        mlflow.log_metric("train80_sec", train_timer.duration)

        with log_exec_timer(
            f"{model_name} prediction"
        ) as infer_timer, JobGroup(
            "Model inference", f"{model2.__class__.__name__}.predict()"
        ):
            if isinstance(model2, AssociationRulesItemRec):
                recs = model2.get_nearest_items(
                    items=test,
                    k=k,
                )
            else:
                recs = model2.predict(
                    k=k,
                    users=test.select("user_idx").distinct(),
                    log=train80,
                    filter_seen_items=filter_seen_items,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer80_sec", infer_timer.duration)

        if not isinstance(model2, AssociationRulesItemRec):
            with log_exec_timer(
                f"Metrics calculation"
            ) as metrics_timer, JobGroup(
                "Metrics calculation", "e.add_result()"
            ):
                e = Experiment(
                    test,
                    {
                        MAP(use_scala_udf=use_scala_udf): k_list_metrics,
                        NDCG(use_scala_udf=use_scala_udf): k_list_metrics,
                        HitRate(use_scala_udf=use_scala_udf): k_list_metrics,
                    },
                )
                e.add_result(model_name, recs)
            mlflow.log_metric("metrics80_sec", metrics_timer.duration)
            metrics = dict()
            for k in k_list_metrics:
                metrics["NDCG.{}_80".format(k)] = e.results.at[
                    model_name, "NDCG@{}".format(k)
                ]
                metrics["MAP.{}_80".format(k)] = e.results.at[
                    model_name, "MAP@{}".format(k)
                ]
                metrics["HitRate.{}_80".format(k)] = e.results.at[
                    model_name, "HitRate@{}".format(k)
                ]
            mlflow.log_metrics(metrics)


if __name__ == "__main__":
    spark_sess = get_spark_session()
    dataset = os.environ.get("DATASET", "ml1m")  # ml1m
    # os.environ["MODEL"] = "UCB"
    main(spark=spark_sess, dataset_name=dataset)
    spark_sess.stop()
