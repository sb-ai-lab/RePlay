import os

import mlflow

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

def get_model(MODEL: str, SEED: int, spark_app_id: str):
    """Inits model and return instance
    """

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
    elif MODEL == "ALS_HNSWLIB":
        ALS_RANK = int(os.environ.get("ALS_RANK", 100))
        build_index_on = "driver"  # driver executor
        num_blocks = int(os.environ.get("NUM_BLOCKS", 10))
        hnswlib_params = {
            "space": "ip",
            "M": 16,
            "efS": 200,
            "efC": 200,
            "post": 0,
            # hdfs://node21.bdcl:9000
            # "index_path": f"/opt/spark_data/replay_datasets/nmslib_hnsw_index_{spark_app_id}",
            "build_index_on": build_index_on,
        }
        mlflow.log_params(
            {
                "ALS_rank": ALS_RANK,
                "num_blocks": num_blocks,
                "build_index_on": build_index_on,
                "hnswlib_params": hnswlib_params,
            }
        )
        model = ALSWrap(
            rank=ALS_RANK,
            seed=SEED,
            num_item_blocks=num_blocks,
            num_user_blocks=num_blocks,
            hnswlib_params=hnswlib_params,
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
        num_neighbours = int(os.environ.get("NUM_NEIGHBOURS", 10))
        mlflow.log_param("num_neighbours", num_neighbours)
        model = ItemKNN(num_neighbours=num_neighbours)
    elif MODEL == "ItemKNN_NMSLIB_HNSW":
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
    elif MODEL == "ClusterRec_HNSWLIB":
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
    elif MODEL == "UCB":
        model = UCB(seed=SEED)
    else:
        raise ValueError("Unknown model.")

    return model