"""
This script is a Spark application that executes replay TwoStagesScenario.
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
    The "space" parameter described on the page https://github.com/nmslib/nmslib/blob/master/manual/spaces.md.
    Parameters "M", "efS" and "efC" are described at  https://github.com/nmslib/nmslib/blob/master/manual/methods.md#graph-based-search-methods-sw-graph-and-hnsw.
    The "build_index_on" parameter specifies whether the index will be built on the "driver" or "executor".
    If "build_index_on"="executor" then "index_path" must be specified.
    "index_path" determines where the built index should be stored. "index_path" can be a path in hdfs.
    If "build_index_on"="driver", then the index built on the driver will be passed to the executors via the `SparkContext.addFile` mechanism.

    HNSWLIB_PARAMS: hnswlib index params. Double quotes must be used instead of single quotes
    Example: {"space":"ip","M":100,"efS":2000,"efC":2000,"post":0,
    "index_path":"/tmp/hnswlib_index_{spark_app_id}","build_index_on":"executor"}
    The "space" parameter described on the page https://github.com/nmslib/hnswlib/blob/master/README.md#supported-distances
    Parameters "M", "efS" and "efC" are described at https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
    Parameters "build_index_on" and "index_path" are the same as for NMSLIB_HNSW_PARAMS

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
from importlib.metadata import version

import mlflow
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

from experiment_utils import (
    get_datasets,
    get_spark_configs_as_dict,
    check_number_of_allocated_executors,
    get_partition_num,
    prepare_datasets, get_models,
)
from replay.experiment import Experiment
from replay.metrics import HitRate, MAP, NDCG
from replay.scenarios import TwoStagesScenario
from replay.session_handler import get_spark_session
from replay.splitters import UserSplitter
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
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
# logger.addHandler(streamHandler)

# fileHandler = logging.FileHandler("/tmp/replay.log")
# fileHandler.setFormatter(formatter)
# logger.addHandler(fileHandler)

# logger.setLevel(logging.DEBUG)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        # fileHandler,
        streamHandler
    ],
)

logging.getLogger("urllib3").setLevel(logging.WARNING)


def main(spark: SparkSession, dataset_name: str):
    spark_conf: SparkConf = spark.sparkContext.getConf()

    check_number_of_allocated_executors(spark)

    k = int(os.environ.get("K", 100))
    k_list_metrics = list(
        map(int, os.environ.get("K_LIST_METRICS", "5,10,25,100").split(","))
    )
    seed = int(os.environ.get("SEED", 1234))
    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8822"
    )
    model_name = os.environ.get("MODEL", "some_mode")

    partition_num = get_partition_num(spark_conf)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(os.environ.get("EXPERIMENT", "delete"))

    with mlflow.start_run():
        params = get_spark_configs_as_dict(spark_conf)
        params.update(
            {
                "spark.applicationId": spark.sparkContext.applicationId,
                "pyspark": version("pyspark"),
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

        first_levels_models_params = {
            "replay.models.knn.ItemKNN": {"num_neighbours": int(os.environ.get("NUM_NEIGHBOURS", 100))},
            "replay.models.als.ALSWrap": {
                "rank": int(os.environ.get("ALS_RANK", 100)),
                "seed": seed,
                "num_item_blocks": int(os.environ.get("NUM_BLOCKS", 10)),
                "num_user_blocks": int(os.environ.get("NUM_BLOCKS", 10)),
                "hnswlib_params": {
                    "space": "ip",
                    "M": 100,
                    "efS": 2000,
                    "efC": 2000,
                    "post": 0,
                    "index_path": f"/tmp/als_hnswlib_index_{spark.sparkContext.applicationId}",
                    "build_index_on": "executor",
                },
            },
            "replay.models.word2vec.Word2VecRec": {
                "rank": int(os.environ.get("WORD2VEC_RANK", 100)),
                "seed": seed,
                "hnswlib_params": {
                    "space": "ip",
                    "M": 100,
                    "efS": 2000,
                    "efC": 2000,
                    "post": 0,
                    "index_path": f"/tmp/word2vec_hnswlib_index_{spark.sparkContext.applicationId}",
                    "build_index_on": "executor",
                },
            },
        }
        mlflow.log_params(first_levels_models_params)

        # Initialization of first level models
        first_level_models = get_models(first_levels_models_params)
        use_first_level_models_feat = [False, True, False]
        assert len(first_level_models) == len(use_first_level_models_feat)

        mlflow.log_param(
            "first_level_models",
            [type(m).__name__ for m in first_level_models],
        )
        mlflow.log_param(
            "use_first_level_models_feat", use_first_level_models_feat
        )

        second_model_params = {
            "cpu_limit": 80,  # 20
            "memory_limit": int(80 * 0.95),  # 40
            "timeout": 400,
            "general_params": {"use_algos": [["lgb"]]},
            "lgb_params": {
                "use_single_dataset_mode": True,
                "convert_to_onnx": False,
                "mini_batch_size": 1000,
            },
            "linear_l2_params": {"default_params": {"regParam": [1e-5]}},
            "reader_params": {"cv": 2, "advanced_roles": False, "samples": 10_000}
        }
        mlflow.log_param("second_model_params", second_model_params)

        scenario = TwoStagesScenario(
            train_splitter=UserSplitter(
                item_test_size=0.2, shuffle=True, seed=42
            ),
            first_level_models=first_level_models,
            use_first_level_models_feat=use_first_level_models_feat,
            second_model_type="slama",
            second_model_params=second_model_params,
            second_model_config_path=os.environ.get(
                "PATH_TO_SLAMA_TABULAR_CONFIG", "tabular_config.yml"
            ),
            use_generated_features=True
        )

        # Model fitting
        with log_exec_timer(
            f"{type(scenario).__name__} fitting"
        ) as timer, JobGroup(
            f"{type(scenario).__name__} fitting",
            f"{type(scenario).__name__}.fit()",
        ):
            scenario.fit(log=train, user_features=None, item_features=None)
        mlflow.log_metric(f"{type(scenario).__name__}.fit_sec", timer.duration)

        # Model inference
        with log_exec_timer(
            f"{type(scenario).__name__} inference"
        ) as timer, JobGroup(
            f"{type(scenario).__name__} inference",
            f"{type(scenario).__name__}.predict()",
        ):
            recs = scenario.predict(
                log=train,
                k=k,
                users=test.select("user_idx").distinct(),
                filter_seen_items=True,
            )
        mlflow.log_metric(
            f"{type(scenario).__name__}.predict_sec", timer.duration
        )

        with log_exec_timer("Metrics calculation") as metrics_timer, JobGroup(
            "Metrics calculation", "e.add_result()"
        ):
            e = Experiment(
                test,
                {
                    MAP(): k_list_metrics,
                    NDCG(): k_list_metrics,
                    HitRate(): k_list_metrics,
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


if __name__ == "__main__":
    spark_sess = get_spark_session()
    dataset = os.environ.get("DATASET", "MovieLens_1m")
    main(spark=spark_sess, dataset_name=dataset)
    spark_sess.stop()
