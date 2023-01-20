import os

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame

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
from replay.utils import log_exec_timer


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
        build_index_on = "executor"  # driver executor
        num_blocks = int(os.environ.get("NUM_BLOCKS", 10))
        hnswlib_params = {
            "space": "ip",
            "M": 100,
            "efS": 2000,
            "efC": 2000,
            "post": 0,
            # hdfs://node21.bdcl:9000
            "index_path": f"/opt/spark_data/replay_datasets/hnsw_index_{spark_app_id}",
            "build_index_on": build_index_on,
        }
        mlflow.log_params(
            {
                "ALS_rank": als_rank,
                "num_blocks": num_blocks,
                "build_index_on": build_index_on,
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
    elif model_name == "ALS_SCANN":
        als_rank = int(os.environ.get("ALS_RANK", 100))
        build_index_on = "executor"  # driver executor
        num_blocks = int(os.environ.get("NUM_BLOCKS", 10))
        scann_params = {
            "distance_measure": "dot_product",
            "num_neighbors": 10,
            # "efS": 2000,
            # "efC": 2000,
            # "post": 0,
            # hdfs://node21.bdcl:9000
            "index_path": f"/opt/spark_data/replay_datasets/scann_index_{spark_app_id}",
            "build_index_on": build_index_on,
        }
        mlflow.log_params(
            {
                "ALS_rank": als_rank,
                "num_blocks": num_blocks,
                "build_index_on": build_index_on,
                "scann_params": scann_params,
            }
        )
        model = ALSWrap(
            rank=als_rank,
            seed=seed,
            num_item_blocks=num_blocks,
            num_user_blocks=num_blocks,
            scann_params=scann_params,
        )
    elif model_name == "SLIM":
        model = SLIM(seed=seed)
    elif model_name == "SLIM_NMSLIB_HNSW":
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
        model = SLIM(seed=seed, nmslib_hnsw_params=nmslib_hnsw_params)
    elif model_name == "ItemKNN":
        num_neighbours = int(os.environ.get("NUM_NEIGHBOURS", 10))
        mlflow.log_param("num_neighbours", num_neighbours)
        model = ItemKNN(num_neighbours=num_neighbours)
    elif model_name == "ItemKNN_NMSLIB_HNSW":
        build_index_on = "executor"  # driver executor
        nmslib_hnsw_params = {
            "method": "hnsw",
            "space": "negdotprod_sparse_fast",  # negdotprod_sparse_fast cosinesimil_sparse negdotprod_sparse
            "M": 16,
            "efS": 200,
            "efC": 200,
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
        model = ItemKNN(nmslib_hnsw_params=nmslib_hnsw_params)
    elif model_name == "LightFM":
        model = LightFMWrap(random_state=seed)
    elif model_name == "Word2VecRec":
        # model = Word2VecRec(
        #     seed=SEED,
        #     num_partitions=partition_num,
        # )
        model = Word2VecRec(seed=seed)
    elif model_name == "Word2VecRec_NMSLIB_HNSW":
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
        word2vec_rank = int(os.environ.get("WORD2VEC_RANK", 100))
        mlflow.log_params(
            {
                "build_index_on": build_index_on,
                "nmslib_hnsw_params": nmslib_hnsw_params,
                "word2vec_rank": word2vec_rank,
            }
        )

        model = Word2VecRec(
            rank=word2vec_rank,
            seed=seed,
            nmslib_hnsw_params=nmslib_hnsw_params,
        )
    elif model_name == "Word2VecRec_HNSWLIB":
        build_index_on = "executor"  # driver executor
        hnswlib_params = {
            "space": "ip",
            "M": 100,
            "efS": 2000,
            "efC": 2000,
            "post": 0,
            # hdfs://node21.bdcl:9000
            "index_path": f"/opt/spark_data/replay_datasets/hnswlib_index_{spark_app_id}",
            "build_index_on": build_index_on,
        }
        word2vec_rank = int(os.environ.get("WORD2VEC_RANK", 100))
        mlflow.log_params(
            {
                "build_index_on": build_index_on,
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
        num_clusters = int(os.environ.get("num_clusters", "10"))
        model = ClusterRec(num_clusters=num_clusters)
    elif model_name == "ClusterRec_HNSWLIB":
        build_index_on = "driver"
        hnswlib_params = {
            "space": "ip",
            "M": 16,
            "efS": 200,
            "efC": 200,
            # hdfs://node21.bdcl:9000
            # "index_path": f"/opt/spark_data/replay_datasets/nmslib_hnsw_index_{spark_app_id}",
            "build_index_on": build_index_on,
        }
        mlflow.log_param("hnswlib_params", hnswlib_params)
        model = ClusterRec(hnswlib_params=hnswlib_params)
    elif model_name == "UCB":
        model = UCB(seed=seed)
    else:
        raise ValueError("Unknown model.")

    return model


def get_datasets(dataset_name, spark: SparkSession, partition_num: int):
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
                f"/opt/spark_data/replay_datasets/MovieLens/train_{dataset_version}.parquet"
            )
            test = spark.read.parquet(  # hdfs://node21.bdcl:9000
                f"/opt/spark_data/replay_datasets/MovieLens/test_{dataset_version}.parquet"
            )
        train = train.repartition(partition_num)
        test = test.repartition(partition_num)
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
            # user_features = spark.read.parquet(
            #     "/opt/spark_data/replay_datasets/ml1m_user_features.parquet"
            # )
            # .select("user_idx", "gender_idx", "age", "occupation", "zip_code_idx")
            train = train.repartition(partition_num, "user_idx")
            test = test.repartition(partition_num, "user_idx")
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
            train = train.repartition(partition_num, "user_idx")
            test = test.repartition(partition_num, "user_idx")
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
            train = train.repartition(partition_num, "user_idx")
            test = test.repartition(partition_num, "user_idx")
    else:
        raise ValueError("Unknown dataset.")

    mlflow.log_metric("parquets_read_sec", parquets_read_timer.duration)

    return train, test, user_features

