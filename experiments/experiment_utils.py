import importlib
import json
import os
from typing import Tuple, Optional, cast, Dict, List

import mlflow
import pandas as pd
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from rs_datasets import MovieLens, MillionSongDataset

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
    UCB,
)
from replay.models.base_rec import BaseRecommender
# from replay.utils import log_exec_timer, get_number_of_allocated_executors
from replay.data_preparator import DataPreparator, Indexer
from replay.splitters import DateSplitter, UserSplitter


def get_nmslib_hnsw_params(spark_app_id: str):
    index_params_str = os.environ.get("NMSLIB_HNSW_PARAMS")
    if not index_params_str:
        raise ValueError(
            f"To use nmslib hnsw index you need to set the 'NMSLIB_HNSW_PARAMS' env variable! "
            'For example, {"method":"hnsw","space":"negdotprod_sparse_fast","M":16,"efS":200,"efC":200,"post":0,'
            '"index_path":"/tmp/nmslib_hnsw_index_{spark_app_id}","build_index_on":"executor"}.'
        )
    nmslib_hnsw_params = json.loads(index_params_str)
    if (
        "index_path" in nmslib_hnsw_params
        and "{spark_app_id}" in nmslib_hnsw_params["index_path"]
    ):
        nmslib_hnsw_params["index_path"] = nmslib_hnsw_params[
            "index_path"
        ].replace("{spark_app_id}", spark_app_id)
    print(f"nmslib_hnsw_params: {nmslib_hnsw_params}")
    return nmslib_hnsw_params


def get_hnswlib_params(spark_app_id: str):
    index_params_str = os.environ.get("HNSWLIB_PARAMS")
    if not index_params_str:
        raise ValueError(
            f"To use hnswlib index you need to set the 'HNSWLIB_PARAMS' env variable! "
            'For example, {"space":"ip","M":100,"efS":2000,"efC":2000,"post":0,'
            '"index_path":"/tmp/hnswlib_index_{spark_app_id}","build_index_on":"executor"}.'
        )
    hnswlib_params = json.loads(index_params_str)
    if (
        "index_path" in hnswlib_params
        and "{spark_app_id}" in hnswlib_params["index_path"]
    ):
        hnswlib_params["index_path"] = hnswlib_params["index_path"].replace(
            "{spark_app_id}", spark_app_id
        )
    print(f"hnswlib_params: {hnswlib_params}")
    return hnswlib_params


def get_model(model_name: str, seed: int, spark_app_id: str):
    """Initializes model and returns an instance of it

    Args:
        model_name: model name indicating which model to use. For example, `ALS` and `ALS_HNSWLIB`, where second is ALS with the hnsw index.
        seed: seed
        spark_app_id: spark application id. used for model artifacts paths.
    """

    if model_name == "ALS":
        als_rank = int(os.environ.get("ALS_RANK", 100))
        num_blocks = int(os.environ.get("NUM_BLOCKS", 10))

        mlflow.log_params({"num_blocks": num_blocks, "ALS_rank": als_rank})

        model = ALSWrap(
            rank=als_rank,
            seed=seed,
            num_item_blocks=num_blocks,
            num_user_blocks=num_blocks,
        )

    elif model_name == "Explicit_ALS":
        als_rank = int(os.environ.get("ALS_RANK", 100))
        mlflow.log_param("ALS_rank", als_rank)
        model = ALSWrap(rank=als_rank, seed=seed, implicit_prefs=False)
    elif model_name == "ALS_HNSWLIB":
        als_rank = int(os.environ.get("ALS_RANK", 100))
        num_blocks = int(os.environ.get("NUM_BLOCKS", 10))
        hnswlib_params = get_hnswlib_params(spark_app_id)
        mlflow.log_params(
            {
                "ALS_rank": als_rank,
                "num_blocks": num_blocks,
                "build_index_on": hnswlib_params["build_index_on"],
                "hnswlib_params": hnswlib_params,
            }
        )
        model = ALSWrap(
            rank=als_rank,
            seed=seed,
            num_item_blocks=num_blocks,
            num_user_blocks=num_blocks,
            hnswlib_params=hnswlib_params,
        )
    elif model_name == "SLIM":
        model = SLIM(seed=seed)
    elif model_name == "SLIM_NMSLIB_HNSW":
        nmslib_hnsw_params = get_nmslib_hnsw_params(spark_app_id)
        mlflow.log_params(
            {
                "build_index_on": nmslib_hnsw_params["build_index_on"],
                "nmslib_hnsw_params": nmslib_hnsw_params,
            }
        )
        model = SLIM(seed=seed, nmslib_hnsw_params=nmslib_hnsw_params)
    elif model_name == "ItemKNN":
        num_neighbours = int(os.environ.get("NUM_NEIGHBOURS", 10))
        mlflow.log_param("num_neighbours", num_neighbours)
        model = ItemKNN(num_neighbours=num_neighbours)
    elif model_name == "ItemKNN_NMSLIB_HNSW":
        num_neighbours = int(os.environ.get("NUM_NEIGHBOURS", 10))
        nmslib_hnsw_params = get_nmslib_hnsw_params(spark_app_id)
        mlflow.log_params(
            {
                "build_index_on": nmslib_hnsw_params["build_index_on"],
                "nmslib_hnsw_params": nmslib_hnsw_params,
                "num_neighbours": num_neighbours
            }
        )
        model = ItemKNN(num_neighbours=num_neighbours, nmslib_hnsw_params=nmslib_hnsw_params)
    elif model_name == "LightFM":
        model = LightFMWrap(random_state=seed)
    elif model_name == "Word2VecRec":
        word2vec_rank = int(os.environ.get("WORD2VEC_RANK", 100))
        mlflow.log_param("word2vec_rank", word2vec_rank)
        model = Word2VecRec(rank=word2vec_rank, seed=seed)
    elif model_name == "Word2VecRec_HNSWLIB":
        hnswlib_params = get_hnswlib_params(spark_app_id)
        word2vec_rank = int(os.environ.get("WORD2VEC_RANK", 100))
        mlflow.log_params(
            {
                "build_index_on": hnswlib_params["build_index_on"],
                "hnswlib_params": hnswlib_params,
                "word2vec_rank": word2vec_rank,
            }
        )

        model = Word2VecRec(
            rank=word2vec_rank,
            seed=seed,
            hnswlib_params=hnswlib_params,
        )
    elif model_name == "PopRec":
        use_relevance = os.environ.get("USE_RELEVANCE", "False") == "True"
        model = PopRec(use_relevance=use_relevance)
        mlflow.log_param("USE_RELEVANCE", use_relevance)
    elif model_name == "UserPopRec":
        model = UserPopRec()
    elif model_name == "RandomRec_uniform":
        model = RandomRec(seed=seed, distribution="uniform")
    elif model_name == "RandomRec_popular_based":
        model = RandomRec(seed=seed, distribution="popular_based")
    elif model_name == "RandomRec_relevance":
        model = RandomRec(seed=seed, distribution="relevance")
    elif model_name == "AssociationRulesItemRec":
        model = AssociationRulesItemRec()
    elif model_name == "Wilson":
        model = Wilson()
    elif model_name == "ClusterRec":
        num_clusters = int(os.environ.get("NUM_CLUSTERS", "10"))
        mlflow.log_param("num_clusters", num_clusters)
        model = ClusterRec(num_clusters=num_clusters)
    elif model_name == "ClusterRec_HNSWLIB":
        num_clusters = int(os.environ.get("NUM_CLUSTERS", "10"))
        hnswlib_params = get_hnswlib_params(spark_app_id)
        mlflow.log_params(
            {
                "num_clusters": num_clusters,
                "build_index_on": hnswlib_params["build_index_on"],
                "hnswlib_params": hnswlib_params,
            }
        )
        model = ClusterRec(
            num_clusters=num_clusters, hnswlib_params=hnswlib_params
        )
    elif model_name == "UCB":
        model = UCB(seed=seed)
    else:
        raise ValueError("Unknown model.")

    return model


def get_models(models: Dict) -> List[BaseRecommender]:

    list_of_models = []
    for model_class_name, model_kwargs in models.items():
        module_name = ".".join(model_class_name.split('.')[:-1])
        class_name = model_class_name.split('.')[-1]
        module = importlib.import_module(module_name)
        clazz = getattr(module, class_name)

        base_model = cast(BaseRecommender, clazz(**model_kwargs))
        list_of_models.append(base_model)

    return list_of_models

def prepare_datasets(dataset_name: str, spark: SparkSession, partition_num: int):

    if dataset_name.startswith("MovieLens"):
        # name__{size} pattern
        dataset_params = dataset_name.split("__")
        if len(dataset_params) == 1:
            dataset_version = "1m"
        elif len(dataset_params) == 2:
            dataset_version = dataset_params[1]
        else:
            raise ValueError("Too many dataset params.")
        train_target_path = f"{os.environ['DATASETS_DIR']}MovieLens/train_{dataset_version}.parquet"
        test_target_path = f"{os.environ['DATASETS_DIR']}MovieLens/test_{dataset_version}.parquet"
    elif dataset_name.startswith("MillionSongDataset"):
        # MillionSongDataset__{fraction} pattern
        dataset_params = dataset_name.split("__")
        if len(dataset_params) == 1:
            fraction = "1.0"
        else:
            fraction = dataset_params[1]
        train_target_path = f"{os.environ['DATASETS_DIR']}MillionSongDataset/fraction_{fraction}_train.parquet"
        test_target_path = f"{os.environ['DATASETS_DIR']}MillionSongDataset/fraction_{fraction}_test.parquet"
    else:
        raise ValueError("Unknown dataset.")

    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    is_exists = fs.exists(spark._jvm.org.apache.hadoop.fs.Path(train_target_path))
    if is_exists and os.environ.get("FORCE_RECREATE_DATASETS", "False") != "True":
        print(f"Path '{train_target_path}' already exists and FORCE_RECREATE_DATASETS != True. "
              "Skipping datasets creation.")
        return

    if dataset_name.startswith("MovieLens"):
        data = MovieLens(
            dataset_version, path=f"{os.environ['RS_DATASETS_DIR']}MovieLens"
        )
        data = data.ratings
        mapping = {
            "user_id": "user_id",
            "item_id": "item_id",
            "relevance": "rating",
            "timestamp": "timestamp",
        }
    elif dataset_name.startswith("MillionSongDataset"):
        if fraction == "1.0":
            data = MillionSongDataset(
                path=f"{os.environ['RS_DATASETS_DIR']}MillionSongDataset"
            )
            data = data.train
        elif fraction == "train_10x_users":
            data = spark.read.parquet(
                "file:///opt/spark_data/replay_datasets/MillionSongDataset/train_10x_users.parquet"
            )
        elif fraction == "train_100m_users_1k_items":
            data = spark.read.parquet(
                "file:///opt/spark_data/replay_datasets/MillionSongDataset/train_100m_users_1k_items.parquet"
            )
        else:
            data = pd.read_csv(f"/opt/spark_data/replay_datasets/MillionSongDataset/train_{fraction}.csv")

        mapping = {
            "user_id": "user_id",
            "item_id": "item_id",
            "relevance": "play_count",
        }

    with log_exec_timer("DataPreparator execution") as preparator_timer:
        preparator = DataPreparator(columns_mapping=mapping)
        log = preparator.transform(data)
        log = log.repartition(partition_num).cache()
        log.write.mode("overwrite").format("noop").save()
    mlflow.log_metric("preparator_sec", preparator_timer.duration)

    mlflow.log_metric("log_num_partitions", log.rdd.getNumPartitions())

    if os.getenv("FILTER_LOG") == "True":
        with log_exec_timer("log filtering") as log_filtering_timer:
            # will consider ratings >= 3 as positive feedback. A positive feedback is treated with relevance = 1
            only_positives_log = log.filter(
                sf.col("relevance") >= 3  # 1
            ).withColumn("relevance", sf.lit(1))
            only_positives_log = only_positives_log.cache()
            only_positives_log.write.mode("overwrite").format("noop").save()
            log = only_positives_log
        mlflow.log_metric("log_filtering_sec", log_filtering_timer.duration)

    with log_exec_timer(
        "log.count() execution"
    ) as log_count_timer:
        log_length = log.count()
    mlflow.log_metric("log_count_sec", log_count_timer.duration)
    mlflow.log_param("log_length", log_length)

    with log_exec_timer("Indexer training") as indexer_fit_timer:
        indexer = Indexer(user_col="user_id", item_col="item_id")
        indexer.fit(
            users=log.select("user_id"), items=log.select("item_id")
        )
    mlflow.log_metric("indexer_fit_sec", indexer_fit_timer.duration)

    with log_exec_timer("Indexer transform") as indexer_transform_timer:
        log_replay = indexer.transform(df=log)
        log_replay = log_replay.cache()
        log_replay.write.mode("overwrite").format("noop").save()
    mlflow.log_metric(
        "indexer_transform_sec", indexer_transform_timer.duration
    )

    with log_exec_timer("Splitter execution") as splitter_timer:
        if dataset_name.startswith("MovieLens"):
            # MovieLens
            train_spl = DateSplitter(
                test_start=0.2,
                drop_cold_items=True,
                drop_cold_users=True,
            )
        else:
            # MillionSongDataset
            train_spl = UserSplitter(
                item_test_size=0.2,
                shuffle=True,
                drop_cold_items=True,
                drop_cold_users=True,
            )
        train, test = train_spl.split(log_replay)

        train = train.cache()
        test = test.cache()
        train.write.mode("overwrite").format("noop").save()
        test.write.mode("overwrite").format("noop").save()
        test = test.repartition(partition_num).cache()
    mlflow.log_metric("splitter_sec", splitter_timer.duration)

    mlflow.log_metric("train_num_partitions", train.rdd.getNumPartitions())
    mlflow.log_metric("test_num_partitions", test.rdd.getNumPartitions())

    with log_exec_timer("Train/test datasets saving to parquet") as parquets_save_timer:
        # WARN: 'fraction' is not fraction of test or train, it is fraction of input dataset.
        train.write.mode('overwrite').parquet(
            train_target_path
        )
        test.write.mode('overwrite').parquet(
            test_target_path
        )
    mlflow.log_metric(f"parquets_write_sec", parquets_save_timer.duration)


def get_datasets(
    dataset_name, spark: SparkSession, partition_num: int
) -> Tuple[DataFrame, DataFrame, Optional[DataFrame]]:
    """
    Reads prepared datasets from hdfs or disk and returns them.

    Args:
        dataset_name: Dataset name with size postfix (optional). For example `MovieLens__10m` or `MovieLens__25m`.
        spark: spark session
        partition_num: Number of partitions in output dataframes.

    Returns:
        train: train dataset
        test: test dataset
        user_features: dataframe with user features (optional)

    """
    user_features = None
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
                f"{os.environ['DATASETS_DIR']}MovieLens/train_{dataset_version}.parquet"
            )
            test = spark.read.parquet(  # hdfs://node21.bdcl:9000
                f"{os.environ['DATASETS_DIR']}MovieLens/test_{dataset_version}.parquet"
            )
    elif dataset_name.startswith("MillionSongDataset"):
        # MillionSongDataset__{fraction} pattern
        dataset_params = dataset_name.split("__")
        if len(dataset_params) == 1:
            fraction = "1.0"
        else:
            fraction = dataset_params[1]

        if fraction == "train_100m_users_1k_items":
            with log_exec_timer(
                "Train/test datasets reading to parquet"
            ) as parquets_read_timer:
                train = spark.read.parquet(
                    f"{os.environ['DATASETS_DIR']}MillionSongDataset/fraction_{fraction}_train.parquet"
                )
                test = spark.read.parquet(
                    f"{os.environ['DATASETS_DIR']}MillionSongDataset/fraction_{fraction}_test.parquet"
                )
        else:
            if partition_num in {6, 12, 24, 48}:
                with log_exec_timer(
                    "Train/test datasets reading to parquet"
                ) as parquets_read_timer:
                    train = spark.read.parquet(
                        f"{os.environ['DATASETS_DIR']}MillionSongDataset/"
                        f"fraction_{fraction}_train_{partition_num}_partition.parquet"
                    )
                    test = spark.read.parquet(
                        f"{os.environ['DATASETS_DIR']}MillionSongDataset/"
                        f"fraction_{fraction}_test_{partition_num}_partition.parquet"
                    )
            else:
                with log_exec_timer(
                    "Train/test datasets reading to parquet"
                ) as parquets_read_timer:
                    train = spark.read.parquet(
                        f"{os.environ['DATASETS_DIR']}MillionSongDataset/"
                        f"fraction_{fraction}_train_24_partition.parquet"
                    )
                    test = spark.read.parquet(
                        f"{os.environ['DATASETS_DIR']}MillionSongDataset/"
                        f"fraction_{fraction}_test_24_partition.parquet"
                    )
    elif dataset_name == "ml1m":
        with log_exec_timer(
            "Train/test/user_features datasets reading to parquet"
        ) as parquets_read_timer:
            train = spark.read.parquet(
                f"{os.environ['DATASETS_DIR']}ml1m_train.parquet"
            )
            test = spark.read.parquet(
                f"{os.environ['DATASETS_DIR']}ml1m_test.parquet"
            )
            # user_features = spark.read.parquet(
            #     "/opt/spark_data/replay_datasets/ml1m_user_features.parquet"
            # )
            # .select("user_idx", "gender_idx", "age", "occupation", "zip_code_idx")
    elif dataset_name == "ml1m_first_level_default":
        with log_exec_timer(
            "Train/test/user_features datasets reading to parquet"
        ) as parquets_read_timer:
            train = spark.read.parquet(
                "file:///opt/spark_data/replay/experiments/ml1m_first_level_default/train.parquet"
            )
            test = spark.read.parquet(
                "file:///opt/spark_data/replay/experiments/ml1m_first_level_default/test.parquet"
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
    else:
        raise ValueError("Unknown dataset.")

    train = train.repartition(partition_num, "user_idx")
    test = test.repartition(partition_num, "user_idx")

    mlflow.log_metric("parquets_read_sec", parquets_read_timer.duration)

    return train, test, user_features


def get_spark_configs_as_dict(spark_conf: SparkConf):
    return {
        "spark.driver.cores": spark_conf.get("spark.driver.cores"),
        "spark.driver.memory": spark_conf.get("spark.driver.memory"),
        "spark.memory.fraction": spark_conf.get("spark.memory.fraction"),
        "spark.executor.cores": spark_conf.get("spark.executor.cores"),
        "spark.executor.memory": spark_conf.get("spark.executor.memory"),
        "spark.executor.instances": spark_conf.get("spark.executor.instances"),
        "spark.sql.shuffle.partitions": spark_conf.get(
            "spark.sql.shuffle.partitions"
        ),
        "spark.default.parallelism": spark_conf.get(
            "spark.default.parallelism"
        ),
    }


def check_number_of_allocated_executors(spark: SparkSession):
    """
    Checks whether enough executors are allocated or not. If not, then throws an exception.

    Args:
        spark: spark session
    """

    if os.environ.get('CHECK_NUMBER_OF_ALLOCATED_EXECUTORS') != "True":
        return

    spark_conf: SparkConf = spark.sparkContext.getConf()

    # if enough executors is not allocated in the cluster mode, then we stop the experiment
    if spark_conf.get("spark.executor.instances"):
        if get_number_of_allocated_executors(spark) < int(
            spark_conf.get("spark.executor.instances")
        ):
            raise Exception("Not enough executors to run experiment!")


def get_partition_num(spark_conf: SparkConf):
    if os.environ.get("PARTITION_NUM"):
        partition_num = int(os.environ.get("PARTITION_NUM"))
    else:
        if spark_conf.get("spark.cores.max") is None:
            partition_num = os.cpu_count()
        else:
            partition_num = int(spark_conf.get("spark.cores.max"))

    return partition_num


def get_log_info(
    log: DataFrame, user_col="user_idx", item_col="item_idx"
) -> Tuple[int, int, int]:
    """
    Basic log statistics

    >>> from replay.session_handler import State
    >>> spark = State().session
    >>> log = spark.createDataFrame([(1, 2), (3, 4), (5, 2)]).toDF("user_idx", "item_idx")
    >>> log.show()
    +--------+--------+
    |user_idx|item_idx|
    +--------+--------+
    |       1|       2|
    |       3|       4|
    |       5|       2|
    +--------+--------+
    <BLANKLINE>
    >>> rows_count, users_count, items_count = get_log_info(log)
    >>> print((rows_count, users_count, items_count))
    (3, 3, 2)

    :param log: interaction log containing ``user_idx`` and ``item_idx``
    :param user_col: name of a columns containing users' identificators
    :param item_col: name of a columns containing items' identificators

    :returns: statistics string
    """
    cnt = log.count()
    user_cnt = log.select(user_col).distinct().count()
    item_cnt = log.select(item_col).distinct().count()
    return cnt, user_cnt, item_cnt
