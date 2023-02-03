"""
This script is a Spark application that executes replay recommendation models.
Parameters sets via environment variables.

launch example:
    $ export DATASET=MovieLens
    $ export MODEL=ALS
    $ export ALS_RANK=100
    $ export SEED=22
    $ export K=10
    $ python experiments/run_experiment.py

or run in one line:
    $ DATASET=MovieLens MODEL=ALS ALS_RANK=100 SEED=22 K=10 K_LIST_METRICS=5,10 python experiments/run_experiment.py

All params:
    DATASET: dataset name
    Available values:
        MovieLens__100k
        MovieLens==MovieLens__1m
        MovieLens__10m
        MovieLens__20m
        MovieLens__25m
        MillionSongDataset

    MODEL: model name
    Available values:
        LightFM
        PopRec
        UserPopRec
        ALS
        Explicit_ALS
        ALS_HNSWLIB
        Word2VecRec
        Word2VecRec_HNSWLIB
        SLIM
        SLIM_NMSLIB_HNSW
        ItemKNN
        ItemKNN_NMSLIB_HNSW
        ClusterRec
        ClusterRec_HNSWLIB
        RandomRec_uniform
        RandomRec_popular_based
        RandomRec_relevance
        AssociationRulesItemRec
        Wilson
        UCB

    SEED: seed

    K: number of desired recommendations per user

    K_LIST_METRICS: List of K values (separated by commas) to calculate metrics. For example, K_LIST_METRICS=5,10.
    It perform NDCG@5, NDCG@10, MAP@5, MAP@10, HitRate@5 and HitRate@10 calculation.

    NMSLIB_HNSW_PARAMS: nmslib hnsw index params. Double quotes must be used instead of single quotes
    Example: {"method":"hnsw","space":"negdotprod_sparse_fast","M":100,"efS":2000,"efC":2000,"post":0,
    "index_path":"/tmp/nmslib_hnsw_index_{spark_app_id}","build_index_on":"executor"}

    HNSWLIB_PARAMS: hnswlib index params. Double quotes must be used instead of single quotes
    Example: {"space":"ip","M":100,"efS":2000,"efC":2000,"post":0,
    "index_path":"/tmp/hnswlib_index_{spark_app_id}","build_index_on":"executor"}

    ALS_RANK: rank for ALS model, i.e. length of ALS factor vectors

    NUM_BLOCKS: num_item_blocks and num_user_blocks values in ALS model. Default: 10.

    WORD2VEC_RANK: rank of Word2Vec model

    NUM_NEIGHBOURS: ItemKNN param

    NUM_CLUSTERS: number of clusters in Cluster model

    USE_SCALA_UDFS_METRICS: if set to "True", then metrics will be calculated via scala UDFs

    USE_BUCKETING: if set to "True", then train and test dataframes will be bucketed

    DATASETS_DIR: where train and test datasets will be stored

    FORCE_RECREATE_DATASETS: if set to "True", then train and test dataframes will be recreated

    RS_DATASETS_DIR: where files will be downloaded by the rs_datasets package

    FILTER_LOG: if set to "True", the log will be filtered by "relevance" >= 1

    CHECK_NUMBER_OF_ALLOCATED_EXECUTORS: If set to "True", then number of allocated executors will be checked.
    And if there are not enough executors, then the program will stop.

    PARTITION_NUM: number of partition to repartition test and train dataframes.

"""

import logging
import os

import mlflow
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

from experiment_utils import (
    get_model,
    get_datasets,
    get_spark_configs_as_dict,
    check_number_of_allocated_executors,
    get_partition_num,
    get_log_info, prepare_datasets,
)
from replay.dataframe_bucketizer import DataframeBucketizer
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

    check_number_of_allocated_executors(spark)

    k = int(os.environ.get("K", 10))
    k_list_metrics = list(map(int, os.environ.get("K_LIST_METRICS", "5,10,25,100").split(",")))
    seed = int(os.environ.get("SEED", 1234))
    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8822"
    )
    model_name = os.environ["MODEL"]

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

        prepare_datasets(dataset_name, spark, partition_num)

        train, test, user_features = get_datasets(
            dataset_name, spark, partition_num
        )

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
                    f"bucketed_train_{spark.sparkContext.applicationId.replace('-', '_')}"
                )
                train = bucketizer.transform(train)

                bucketizer.set_table_name(
                    f"bucketed_test_{spark.sparkContext.applicationId.replace('-', '_')}"
                )
                test = bucketizer.transform(test)
            mlflow.log_metric("bucketing_sec", bucketing_timer.duration)

        with log_exec_timer("Train/test caching") as train_test_cache_timer:
            train = train.cache()
            test = test.cache()
            train.write.mode("overwrite").format("noop").save()
            test.write.mode("overwrite").format("noop").save()
        mlflow.log_metric(
            "train_test_cache_sec", train_test_cache_timer.duration
        )

        if model_name in {"UCB", "Wilson"}:
            train = train.withColumn("relevance", sf.when(sf.col("relevance") > 0.0, sf.lit(1)).otherwise(sf.lit(0)))
            test = test.withColumn("relevance", sf.when(sf.col("relevance") > 0.0, sf.lit(1)).otherwise(sf.lit(0)))

        mlflow.log_metric("train_num_partitions", train.rdd.getNumPartitions())
        mlflow.log_metric("test_num_partitions", test.rdd.getNumPartitions())

        with log_exec_timer("get_log_info() execution") as get_log_info_timer:
            (
                train_rows_count,
                train_users_count,
                train_items_count,
            ) = get_log_info(train)
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

        kwargs = {}
        if isinstance(model, ClusterRec):
            kwargs = {"user_features": user_features}

        with log_exec_timer(f"{model_name} training") as train_timer, JobGroup(
            "Model training", f"{model.__class__.__name__}.fit()"
        ):
            model.fit(log=train, **kwargs)
        mlflow.log_metric("train_sec", train_timer.duration)

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
                    log=train,
                    filter_seen_items=True,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer_sec", infer_timer.duration)

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
                        MAP(use_scala_udf=use_scala_udf): k_list_metrics,
                        NDCG(use_scala_udf=use_scala_udf): k_list_metrics,
                        HitRate(use_scala_udf=use_scala_udf): k_list_metrics,
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
    dataset = os.environ.get("DATASET", "MovieLens_1m")
    os.environ['DATASETS_DIR'] = "/opt/spark_data/replay_datasets/MovieLens"
    os.environ['RS_DATASETS_DIR'] = "/opt/spark_data/replay_datasets/MovieLens"
    os.environ["MODEL"] = "PopRec"
    os.environ["MLFLOW_TRACKING_URI"] = "http://node2.bdcl:8811"
    os.environ["HNSWLIB_PARAMS"] = '{"space":"ip","M":100,"efS":2000,"efC":2000,"post":0,"index_path":"/tmp/hnswlib_index_{spark_app_id}","build_index_on":"executor"}'
    os.environ["USE_SCALA_UDFS_METRICS"] = "False"
    main(spark=spark_sess, dataset_name=dataset)
    spark_sess.stop()
