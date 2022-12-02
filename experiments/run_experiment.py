import os
import logging.config
import time

import mlflow
import pandas as pd
from pyspark.sql import SparkSession
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
# logger.setLevel(logging.DEBUG)


def main(spark: SparkSession, dataset_name: str):
    spark_conf: SparkConf = spark.sparkContext.getConf()

    if spark_conf.get("spark.executor.instances"):
        if getNumberOfAllocatedExecutors(spark) < int(
            spark_conf.get("spark.executor.instances")
        ):
            raise Exception("Not enough executors to run experiment!")

    K = int(os.environ.get("K", 10))
    K_list_metrics = [5, 10]
    SEED = int(os.environ.get("SEED", 1234))
    MLFLOW_TRACKING_URI = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8811"
    )
    MODEL = os.environ.get("MODEL", "ItemKNN_NMSLIB_HNSW")
    # PopRec
    # Word2VecRec Word2VecRec_NMSLIB_HNSW
    # ALS ALS_NMSLIB_HNSW
    # SLIM SLIM_NMSLIB_HNSW
    # ItemKNN ItemKNN_NMSLIB_HNSW
    # ClusterRec

    if os.environ.get("PARTITION_NUM"):
        partition_num = int(os.environ.get("PARTITION_NUM"))
    else:
        if spark_conf.get("spark.cores.max") is None:
            partition_num = os.cpu_count()
        else:
            partition_num = int(spark_conf.get("spark.cores.max"))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(
        os.environ.get("EXPERIMENT", "delete")
    )  # os.environ["EXPERIMENT"]

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

            # data = MillionSongDataset(
            #     path="/opt/spark_data/replay_datasets/MillionSongDataset"
            # )
            # data = pd.concat([data.train, data.test, data.val])

            # data = pd.read_csv(f"/opt/spark_data/replay_datasets/MillionSongDataset/train_{fraction}.csv")

            if partition_num in {6, 12, 24, 48}:
                with log_exec_timer(
                    "Train/test datasets reading to parquet"
                ) as parquets_read_timer:
                    train = spark.read.parquet(
                        f"/opt/spark_data/replay_datasets/MillionSongDataset/fraction_{fraction}_train_{partition_num}_partition.parquet"
                    )
                    test = spark.read.parquet(
                        f"/opt/spark_data/replay_datasets/MillionSongDataset/fraction_{fraction}_test_{partition_num}_partition.parquet"
                    )
            else:
                with log_exec_timer(
                    "Train/test datasets reading to parquet"
                ) as parquets_read_timer:
                    train = spark.read.parquet(
                        f"/opt/spark_data/replay_datasets/MillionSongDataset/fraction_{fraction}_train_24_partition.parquet"
                    )
                    test = spark.read.parquet(
                        f"/opt/spark_data/replay_datasets/MillionSongDataset/fraction_{fraction}_test_24_partition.parquet"
                    )
                    train = train.repartition(partition_num)
                    test = test.repartition(partition_num)
            mlflow.log_metric(
                "parquets_read_sec", parquets_read_timer.duration
            )

            # mapping = {
            #     "user_id": "user_id",
            #     "item_id": "item_id",
            #     "relevance": "play_count",
            # }
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
                # .select("user_idx", "gender_idx", "age", "occupation", "zip_code_idx")
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
                    "/opt/spark_data/replay_datasets/ml1m_1m_users_3_7k_items_train.parquet"
                )
                test = spark.read.parquet(
                    "/opt/spark_data/replay_datasets/ml1m_1m_users_3_7k_items_test.parquet"
                )
                user_features = spark.read.parquet(
                    "/opt/spark_data/replay_datasets/ml1m_1m_users_3_7k_items_user_features.parquet"
                )
                # .select("user_idx", "gender_idx", "age", "occupation", "zip_code_idx")
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
                # .select("user_idx", "gender_idx", "age", "occupation", "zip_code_idx")
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

                # logger.debug(
                #     f"spark.catalog.listDatabases(): {str(spark.catalog.listDatabases())}"
                # )
                # logger.debug(
                #     f"spark.catalog.currentDatabase(): {spark.catalog.currentDatabase()}"
                # )
                # logger.debug(
                #     f"spark.catalog.listTables('default'): {str(spark.catalog.listTables('default'))}"
                # )

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
        mlflow.log_metric("get_log_info_sec", get_log_info2_timer.duration)
        mlflow.log_param("train_size", train_info[0])
        mlflow.log_param("train.total_users", train_info[1])
        mlflow.log_param("train.total_items", train_info[2])
        mlflow.log_param("test_size", test_info[0])
        mlflow.log_param("test.total_users", test_info[1])
        mlflow.log_param("test.total_items", test_info[2])

        mlflow.log_param("model", MODEL)
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
                "M": 100,
                "efS": 2000,
                "efC": 2000,
                "post": 0,
                # "index_path": f"/opt/spark_data/replay_datasets/nmslib_hnsw_index_{spark.sparkContext.applicationId}",
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
                hnswlib_params=nmslib_hnsw_params,
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
                "index_path": f"/opt/spark_data/replay_datasets/nmslib_hnsw_index_{spark.sparkContext.applicationId}",
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
            num_neighbours = int(os.environ.get("NUM_NEIGHBOURS", 10))
            mlflow.log_param("num_neighbours", num_neighbours)
            model = ItemKNN(num_neighbours=num_neighbours)
        elif MODEL == "ItemKNN_NMSLIB_HNSW":
            build_index_on = "executor"  # driver executor
            nmslib_hnsw_params = {
                "method": "hnsw",
                "space": "negdotprod_sparse_fast",  # cosinesimil_sparse negdotprod_sparse
                "M": 16,
                "efS": 200,
                "efC": 200,
                "post": 0,
                "index_path": f"/opt/spark_data/replay_datasets/nmslib_hnsw_index_{spark.sparkContext.applicationId}",
                "build_index_on": build_index_on,
            }
            mlflow.log_params(
                {
                    "build_index_on": build_index_on,
                    "nmslib_hnsw_params": nmslib_hnsw_params,
                }
            )
            model = ItemKNN(nmslib_hnsw_params=nmslib_hnsw_params)
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
                "index_path": f"/opt/spark_data/replay_datasets/nmslib_hnsw_index_{spark.sparkContext.applicationId}",
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
            model = PopRec()
        elif MODEL == "UserPopRec":
            model = UserPopRec()
        elif MODEL == "RandomRec_uniform":
            model = RandomRec(seed=SEED, distribution="uniform")
        elif MODEL == "RandomRec_popular_based":
            model = RandomRec(seed=SEED, distribution="popular_based")
        elif MODEL == "AssociationRulesItemRec":
            model = AssociationRulesItemRec()
        elif MODEL == "Wilson":
            model = Wilson()
        elif MODEL == "ClusterRec":
            model = ClusterRec()
        else:
            raise ValueError("Unknown model.")

        kwargs = {}
        if isinstance(model, (ClusterRec)):
            kwargs = {"user_features": user_features}

        with log_exec_timer(f"{MODEL} training") as train_timer, JobGroup(
            "Model training", f"{model.__class__.__name__}.fit()"
        ):
            model.fit(log=train, **kwargs)
        mlflow.log_metric("train_sec", train_timer.duration)

        with log_exec_timer(f"{MODEL} prediction") as infer_timer, JobGroup(
            "Model inference", f"{model.__class__.__name__}.predict()"
        ):
            recs = model.predict(
                k=K,
                users=test.select("user_idx").distinct(),
                log=train,
                filter_seen_items=True,
                **kwargs,
            )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer_sec", infer_timer.duration)

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

        with log_exec_timer(f"Model saving") as model_save_timer:
            save(
                model,
                path=f"/tmp/replay/{MODEL}_{dataset_name}_{spark.sparkContext.applicationId}",  # file://
                overwrite=True,
            )
        mlflow.log_param(
            "model_save_dir",
            f"/tmp/replay/{MODEL}_{dataset_name}_{spark.sparkContext.applicationId}",
        )
        mlflow.log_metric("model_save_sec", model_save_timer.duration)

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
    # dataset = "MovieLens__1m"
    # dataset = "MillionSongDataset"
    main(spark=spark_sess, dataset_name=dataset)
    # time.sleep(100)
    spark_sess.stop()
