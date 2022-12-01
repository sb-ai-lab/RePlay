import os
import logging.config
import time

import mlflow
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, IntegerType, StringType, DateType, LongType
from replay.experiment import Experiment
from replay.metrics import HitRate, MAP, NDCG
from replay.model_handler import load, save
from replay.session_handler import get_spark_session
from replay.utils import (
    JobGroup,
    getNumberOfAllocatedExecutors,
    log_exec_timer,
)

from replay.models import (
    ALSWrap,
    SLIM,
    LightFMWrap,
    ItemKNN,
    Word2VecRec,
    PopRec,
    RandomRec,
    AssociationRulesItemRec,
    UserPopRec,
    Wilson,
    ClusterRec,
    UCB
)
from replay.utils import logger

# from rs_datasets import MovieLens, MillionSongDataset
from pyspark.sql import functions as sf

from replay.splitters import DateSplitter, UserSplitter
from replay.utils import get_log_info2
from replay.filters import filter_by_min_count, filter_out_low_ratings
from pyspark.conf import SparkConf


# VERBOSE_LOGGING_FORMAT = (
#     "%(asctime)s %(levelname)s %(module)s %(filename)s:%(lineno)d %(message)s"
# )
# logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
# logger = logging.getLogger("replay")
logger.setLevel(logging.DEBUG)

def get_model(MODEL: str, SEED: int, spark_app_id: str):
    if MODEL == "ALS":
        ALS_RANK = int(os.environ.get("ALS_RANK", 100))
        num_blocks = int(os.environ.get("NUM_BLOCKS", 10))

        mlflow.log_params({"num_blocks": num_blocks, "ALS_rank": ALS_RANK})

        model = ALSWrap(
            rank=ALS_RANK,
            seed=SEED,
            num_item_blocks=num_blocks,
            num_user_blocks=num_blocks,
        )

    elif MODEL == "Explicit_ALS":
        ALS_RANK = int(os.environ.get("ALS_RANK", 100))
        mlflow.log_param("ALS_rank", ALS_RANK)
        model = ALSWrap(rank=ALS_RANK, seed=SEED, implicit_prefs=False)
    elif MODEL == "ALS_NMSLIB_HNSW":
        ALS_RANK = int(os.environ.get("ALS_RANK", 100))
        build_index_on = "driver"  # driver executor
        num_blocks = int(os.environ.get("NUM_BLOCKS", 10))
        nmslib_hnsw_params = {
            "method": "hnsw",
            "space": "negdotprod",
            "M": 16,
            "efS": 200,
            "efC": 200,
            "post": 0,
            # "index_path": f"/opt/spark_data/replay_datasets/nmslib_hnsw_index_{spark_app_id}",
            "build_index_on": build_index_on,
        }
        mlflow.log_params(
            {
                "ALS_rank": ALS_RANK,
                "num_blocks": num_blocks,
                "build_index_on": build_index_on,
                "nmslib_hnsw_params": nmslib_hnsw_params,
            }
        )
        model = ALSWrap(
            rank=ALS_RANK,
            seed=SEED,
            num_item_blocks=num_blocks,
            num_user_blocks=num_blocks,
            nmslib_hnsw_params=nmslib_hnsw_params,
        )
    elif MODEL == "SLIM":
        model = SLIM(seed=SEED)
    elif MODEL == "SLIM_NMSLIB_HNSW":
        build_index_on = "executor"  # driver executor
        nmslib_hnsw_params = {
            "method": "hnsw",
            "space": "negdotprod_sparse",  # cosinesimil_sparse negdotprod_sparse
            "M": 100,
            "efS": 2000,
            "efC": 2000,
            "post": 0,
            "index_path": f"/opt/spark_data/replay_datasets/nmslib_hnsw_index_{spark_app_id}",
            "build_index_on": build_index_on,
        }
        mlflow.log_params(
            {
                "build_index_on": build_index_on,
                "nmslib_hnsw_params": nmslib_hnsw_params,
            }
        )
        model = SLIM(seed=SEED, nmslib_hnsw_params=nmslib_hnsw_params)
    elif MODEL == "ItemKNN":
        model = ItemKNN(num_neighbours=100)
    elif MODEL == "ItemKNN_NMSLIB_HNSW":
        build_index_on = "executor"  # driver executor
        nmslib_hnsw_params = {
            "method": "hnsw",
            "space": "negdotprod_sparse",  # cosinesimil_sparse negdotprod_sparse
            "M": 100,
            "efS": 2000,
            "efC": 2000,
            "post": 0,
            "index_path": f"/opt/spark_data/replay_datasets/nmslib_hnsw_index_{spark_app_id}",
            "build_index_on": build_index_on,
        }
        mlflow.log_params(
            {
                "build_index_on": build_index_on,
                "nmslib_hnsw_params": nmslib_hnsw_params,
            }
        )
        model = ItemKNN(
            nmslib_hnsw_params=nmslib_hnsw_params
        )
    elif MODEL == "LightFM":
        model = LightFMWrap(random_state=SEED)
    elif MODEL == "Word2VecRec":
        # model = Word2VecRec(
        #     seed=SEED,
        #     num_partitions=partition_num,
        # )
        model = Word2VecRec(seed=SEED)
    elif MODEL == "Word2VecRec_NMSLIB_HNSW":
        build_index_on = "executor"  # driver executor
        nmslib_hnsw_params = {
            "method": "hnsw",
            "space": "negdotprod",
            "M": 100,
            "efS": 2000,
            "efC": 2000,
            "post": 0,
            # hdfs://node21.bdcl:9000
            "index_path": f"/opt/spark_data/replay_datasets/nmslib_hnsw_index_{spark_app_id}",
            "build_index_on": build_index_on,
        }
        mlflow.log_params(
            {
                "build_index_on": build_index_on,
                "nmslib_hnsw_params": nmslib_hnsw_params,
            }
        )

        model = Word2VecRec(
            seed=SEED,
            nmslib_hnsw_params=nmslib_hnsw_params,
        )
    elif MODEL == "PopRec":
        use_relevance = os.environ.get("USE_RELEVANCE", "False") == "True"
        model = PopRec(use_relevance=use_relevance)
        mlflow.log_param("USE_RELEVANCE", use_relevance)
    elif MODEL == "UserPopRec":
        model = UserPopRec()
    elif MODEL == "RandomRec_uniform":
        model = RandomRec(seed=SEED, distribution="uniform")
    elif MODEL == "RandomRec_popular_based":
        model = RandomRec(seed=SEED, distribution="popular_based")
    elif MODEL == "RandomRec_relevance":
        model = RandomRec(seed=SEED, distribution="relevance")
    elif MODEL == "AssociationRulesItemRec":
        model = AssociationRulesItemRec()
    elif MODEL == "Wilson":
        model = Wilson()
    elif MODEL == "ClusterRec":
        model = ClusterRec()
    elif MODEL == "UCB":
        model = UCB(seed=SEED)
    else:
        raise ValueError("Unknown model.")

    return model


def main(spark: SparkSession, dataset_name: str):
    spark_conf: SparkConf = spark.sparkContext.getConf()

    if spark_conf.get("spark.executor.instances"):
        if getNumberOfAllocatedExecutors(spark) < int(
            spark_conf.get("spark.executor.instances")
        ):
            raise Exception("Not enough executors to run experiment!")

    K = int(os.environ.get("K", 10))
    K_list_metrics = [10]
    SEED = int(os.environ.get("SEED", 1234))
    MLFLOW_TRACKING_URI = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8811"
    )
    MODEL = os.environ.get("MODEL", "ALS_NMSLIB_HNSW")
    # PopRec
    # Word2VecRec Word2VecRec_NMSLIB_HNSW
    # ALS ALS_NMSLIB_HNSW
    # SLIM SLIM_NMSLIB_HNSW
    # ItemKNN ItemKNN_NMSLIB_HNSW
    # ClusterRec
    # UCB

    if os.environ.get("PARTITION_NUM"):
        partition_num = int(os.environ.get("PARTITION_NUM"))
    else:
        if spark_conf.get("spark.cores.max") is None:
            partition_num = os.cpu_count()
        else:
            partition_num = int(spark_conf.get("spark.cores.max"))  # 28

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(
        os.environ.get("EXPERIMENT", "delete")
    )  # os.environ["EXPERIMENT"]

    with mlflow.start_run():

        params = {
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
                    # "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/train70.csv"
                    "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/train70_ml1m.csv"
                )
            )
            train80 = (
                spark.read.option("header", True)
                .format("csv")
                .schema(schema)
                .load(
                    # "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/train80.csv"
                    "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/train80_ml1m.csv"
                )
            )
            train_diff80 = (
                spark.read.option("header", True)
                .format("csv")
                .schema(schema)
                .load(
                    # "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/train_dif80.csv"
                    "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/train_dif80_ml1m.csv"
                )
            )
            test = (
                spark.read.option("header", True)
                .format("csv")
                .schema(schema)
                .load(
                    # "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/test.csv"
                    "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/test_ml1m.csv"
                )
            )
        elif dataset_name == "MillionSongDataset":
            # schema = (
            #     StructType()
            #     .add("relevance", IntegerType(), True)
            #     .add("timestamp", DateType(), True)
            #     .add("user_idx", StringType(), True)
            #     .add("item_idx", StringType(), True)
            # )

            train70 = (
                spark.read.parquet(
                    "/opt/spark_data/replay_datasets/MillionSongDataset/train70.parquet"
                )
            )
            train80 = (
                spark.read.parquet(
                    "/opt/spark_data/replay_datasets/MillionSongDataset/train80.parquet"
                )
            )
            train_diff80 = (
                spark.read.parquet(
                    "/opt/spark_data/replay_datasets/MillionSongDataset/train_diff80.parquet"
                )
            )
            test = (
                spark.read.parquet(
                    "/opt/spark_data/replay_datasets/MillionSongDataset/test.parquet"
                )
            )
        else:
            ValueError("Unknown dataset.")

        kwargs = {}
        if MODEL == "ClusterRec":
            user_features = spark.read.parquet(
                "file:///opt/spark_data/replay_datasets/MovieLens/train80_ml1m_user_features.parquet"
            )
            kwargs = {"user_features": user_features}

        mlflow.log_param(
            "USE_BUCKETING", os.environ.get("USE_BUCKETING", "False")
        )
        if os.environ.get("USE_BUCKETING", "False") == "True":
            BUCKETING_KEY = "user_idx"

            with log_exec_timer("Train/test caching") as bucketing_timer:
                (
                    train70.repartition(partition_num, BUCKETING_KEY)
                    .write.mode("overwrite")
                    .bucketBy(partition_num, BUCKETING_KEY)
                    .sortBy(BUCKETING_KEY)
                    .saveAsTable(
                        f"bucketed_train_{spark.sparkContext.applicationId}",
                        format="parquet",
                        path=f"/spark-warehouse/bucketed_train_{spark.sparkContext.applicationId}",
                    )
                )

                train70 = spark.table(
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

        with log_exec_timer(
            "get_log_info2() execution"
        ) as get_log_info2_timer:
            train_info = get_log_info2(train80)
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
        mlflow.log_metric("get_log_info_sec", get_log_info2_timer.duration)
        mlflow.log_param("train_size", train_info[0])
        mlflow.log_param("train.total_users", train_info[1])
        mlflow.log_param("train.total_items", train_info[2])
        mlflow.log_param("test_size", test_info[0])
        mlflow.log_param("test.total_users", test_info[1])
        mlflow.log_param("test.total_items", test_info[2])

        mlflow.log_param("model", MODEL)
        model = get_model(MODEL, SEED, spark.sparkContext.applicationId)

        # kwargs = {}
        # if isinstance(model, (ClusterRec)):
        #     kwargs = {"user_features": user_features}

        filter_seen_items=True
        if isinstance(model, (UserPopRec)):
            filter_seen_items=False

        # logger.setLevel(logging.DEBUG)
        # logger.debug("dataset: train70")
        print("\ndataset: train70")
        with log_exec_timer(f"{MODEL} training") as train_timer, JobGroup(
            "Model training", f"{model.__class__.__name__}.fit()"
        ):
            model.fit(log=train70, **kwargs)
        mlflow.log_metric("train70_sec", train_timer.duration)

        with log_exec_timer(f"{MODEL} prediction") as infer_timer, JobGroup(
            "Model inference", f"{model.__class__.__name__}.predict()"
        ):
            if isinstance(model, (AssociationRulesItemRec)):
                recs = model.get_nearest_items(
                    items=test,
                    k=K,
                )
            else:
                recs = model.predict(
                    k=K,
                    users=test.select("user_idx").distinct(),
                    log=train70,
                    filter_seen_items=filter_seen_items,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer70_sec", infer_timer.duration)

        with log_exec_timer(f"Metrics calculation") as metrics_timer, JobGroup(
            "Metrics calculation", "e.add_result()"
        ):
            e = Experiment(
                test,
                {
                    MAP(): K_list_metrics,
                    NDCG(): K_list_metrics,
                    HitRate(): K_list_metrics,
                },
            )
            e.add_result(MODEL, recs)
        mlflow.log_metric("metrics70_sec", metrics_timer.duration)
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


        print("\ndataset: train_diff")
        with log_exec_timer(
            f"{MODEL} training (additional)"
        ) as train_timer, JobGroup(
            "Model training (additional)", f"{model.__class__.__name__}.fit()"
        ):
            model.refit(log=train_diff80, previous_log=train70)
        mlflow.log_metric("train_diff_sec", train_timer.duration)

        with log_exec_timer(f"{MODEL} prediction") as infer_timer, JobGroup(
            "Model inference", f"{model.__class__.__name__}.predict()"
        ):
            if isinstance(model, (AssociationRulesItemRec)):
                recs = model.get_nearest_items(
                    items=test,
                    k=K,
                )
            else:
                recs = model.predict(
                    k=K,
                    users=test.select("user_idx").distinct(),
                    log=train80, # train_diff80 train80
                    filter_seen_items=filter_seen_items,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer_diff_sec", infer_timer.duration)

        with log_exec_timer(f"Metrics calculation") as metrics_timer, JobGroup(
            "Metrics calculation", "e.add_result()"
        ):
            e = Experiment(
                test,
                {
                    MAP(): K_list_metrics,
                    NDCG(): K_list_metrics,
                    HitRate(): K_list_metrics,
                },
            )
            e.add_result(MODEL, recs)
        mlflow.log_metric("metrics_diff_sec", metrics_timer.duration)
        for k in K_list_metrics:
            mlflow.log_metric(
                "NDCG.{}_diff".format(k),
                e.results.at[MODEL, "NDCG@{}".format(k)],
            )
            mlflow.log_metric(
                "MAP.{}_diff".format(k),
                e.results.at[MODEL, "MAP@{}".format(k)],
            )
            mlflow.log_metric(
                "HitRate.{}_diff".format(k),
                e.results.at[MODEL, "HitRate@{}".format(k)],
            )


        print("\ndataset: train80")
        del model
        model = get_model(MODEL, SEED, spark.sparkContext.applicationId)
        with log_exec_timer(f"{MODEL} training") as train_timer, JobGroup(
            "Model training", f"{model.__class__.__name__}.fit()"
        ):
            model.fit(log=train80, **kwargs)
        mlflow.log_metric("train80_sec", train_timer.duration)

        with log_exec_timer(f"{MODEL} prediction") as infer_timer, JobGroup(
            "Model inference", f"{model.__class__.__name__}.predict()"
        ):
            if isinstance(model, (AssociationRulesItemRec)):
                recs = model.get_nearest_items(
                    items=test,
                    k=K,
                )
            else:
                recs = model.predict(
                    k=K,
                    users=test.select("user_idx").distinct(),
                    log=train80,
                    filter_seen_items=filter_seen_items,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer80_sec", infer_timer.duration)

        with log_exec_timer(f"Metrics calculation") as metrics_timer, JobGroup(
            "Metrics calculation", "e.add_result()"
        ):
            e = Experiment(
                test,
                {
                    MAP(): K_list_metrics,
                    NDCG(): K_list_metrics,
                    HitRate(): K_list_metrics,
                },
            )
            e.add_result(MODEL, recs)
        mlflow.log_metric("metrics80_sec", metrics_timer.duration)
        metrics = dict()
        for k in K_list_metrics:
            metrics["NDCG.{}_80".format(k)] = e.results.at[
                MODEL, "NDCG@{}".format(k)
            ]
            metrics["MAP.{}_80".format(k)] = e.results.at[
                MODEL, "MAP@{}".format(k)
            ]
            metrics["HitRate.{}_80".format(k)] = e.results.at[
                MODEL, "HitRate@{}".format(k)
            ]
        mlflow.log_metrics(metrics)

        # with log_exec_timer(f"Model saving") as model_save_timer:
        #     # save_indexer(indexer, './indexer_ml1')
        #     save(
        #         model,
        #         path=f"/opt/spark_data/replay_datasets/{MODEL}_{dataset_name}", # file://
        #         overwrite=True
        #     )
        # mlflow.log_metric("model_save_sec", model_save_timer.duration)

        # with log_exec_timer(f"Model loading") as model_load_timer:
        #     # save_indexer(indexer, './indexer_ml1')
        #     model_loaded = load(
        #         path=f"/opt/spark_data/replay_datasets/{MODEL}_{dataset_name}"
        #     )
        # mlflow.log_metric("_loaded_model_sec", model_load_timer.duration)

        # with log_exec_timer(f"{MODEL} prediction from loaded model") as infer_loaded_timer:
        #     recs = model_loaded.predict(
        #         k=K,
        #         users=test.select("user_idx").distinct(),
        #         log=train,
        #         filter_seen_items=True,
        #     )
        #     recs.write.mode("overwrite").format("noop").save()
        # mlflow.log_metric("_loaded_infer_sec", infer_loaded_timer.duration)

        # with log_exec_timer(f"Metrics calculation for loaded model") as metrics_loaded_timer:
        #     e = Experiment(
        #         test,
        #         {
        #             MAP(): K_list_metrics,
        #             NDCG(): K_list_metrics,
        #             HitRate(): K_list_metrics,
        #         },
        #     )
        #     e.add_result(MODEL, recs)
        # mlflow.log_metric("_loaded_metrics_sec", metrics_loaded_timer.duration)
        # for k in K_list_metrics:
        #     mlflow.log_metric(
        #         "_loaded_NDCG.{}".format(k), e.results.at[MODEL, "NDCG@{}".format(k)]
        #     )
        #     mlflow.log_metric(
        #         "_loaded_MAP.{}".format(k), e.results.at[MODEL, "MAP@{}".format(k)]
        #     )
        #     mlflow.log_metric(
        #         "_loaded_HitRate.{}".format(k),
        #         e.results.at[MODEL, "HitRate@{}".format(k)],
        #     )


if __name__ == "__main__":
    spark_sess = get_spark_session()
    dataset = os.environ.get("DATASET", "ml1m")  # ml1m
    main(spark=spark_sess, dataset_name=dataset)
    spark_sess.stop()
