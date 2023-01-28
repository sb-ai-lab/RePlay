#!/usr/bin/env bash

set -ex

# Path to python on cluster
PYSPARK_PYTHON_PATH="/python_envs/.replay_venv/bin/python"

# Number of cores to use for the driver process, only in cluster mode (see spark.driver.cores)
DRIVER_CORES="2"

# Amount of memory to use for the driver process (see spark.driver.memory)
DRIVER_MEMORY="20g"

# Limit of total size of serialized results of all partitions for each Spark action (e.g. collect) in bytes.
# (see spark.driver.maxResultSize)
DRIVER_MAX_RESULT_SIZE="5g"

# The number of cores to use on each executor (see spark.executor.cores)
EXECUTOR_CORES="6"

# Amount of memory to use per executor process (see spark.executor.memory)
EXECUTOR_MEMORY="46g"

# Python script to execution on cluster
SCRIPT="run_experiment.py"

# Seed
SEED="42"

# number of desired recommendations per user
K="10"

#
K_LIST_METRICS="5,10"

# num_item_blocks and num_user_blocks values in ALS model
NUM_BLOCKS="10"

# if set to "True", then train and test dataframes will be bucketed
USE_BUCKETING="True"

# rank in ALS model
ALS_RANK="100" # 100

# ItemKNN param (default num_neighbours=10)
NUM_NEIGHBOURS="1000"

# Number of clusters in ClusterRec model
NUM_CLUSTERS="100"

# rank in Word2Vec model
WORD2VEC_RANK="100"

# if set to "True", then metrics will be calculated via scala UDFs
USE_SCALA_UDFS_METRICS="True"

HNSWLIB_PARAMS=\\\{\\\\\\\"space\\\\\\\":\\\\\\\"ip\\\\\\\",\\\\\\\"M\\\\\\\":100,\\\\\\\"efS\\\\\\\":2000,\\\\\\\"efC\\\\\\\":2000,\\\\\\\"post\\\\\\\":0,\\\\\\\"index_path\\\\\\\":\\\\\\\"hdfs://node21.bdcl:9000/tmp/replay/hnswlib_index_\\\{spark_app_id\\\}\\\\\\\",\\\\\\\"build_index_on\\\\\\\":\\\\\\\"executor\\\\\\\"\\\}
NMSLIB_HNSW_PARAMS=\\\{\\\\\\\"method\\\\\\\":\\\\\\\"hnsw\\\\\\\",\\\\\\\"space\\\\\\\":\\\\\\\"negdotprod_sparse_fast\\\\\\\",\\\\\\\"M\\\\\\\":100,\\\\\\\"efS\\\\\\\":2000,\\\\\\\"efC\\\\\\\":2000,\\\\\\\"post\\\\\\\":0,\\\\\\\"index_path\\\\\\\":\\\\\\\"hdfs://node21.bdcl:9000/tmp/replay/nmslib_hnsw_index_\\\{spark_app_id\\\}\\\\\\\",\\\\\\\"build_index_on\\\\\\\":\\\\\\\"executor\\\\\\\"\\\}

DATASETS_DIR="/opt/spark_data/replay_datasets/"
RS_DATASETS_DIR="/opt/spark_data/replay_datasets/"
FORCE_RECREATE_DATASETS="False"
CHECK_NUMBER_OF_ALLOCATED_EXECUTORS="True"


# Dataset name. List of available datasets presented in description of run_experiment.py
for dataset in MovieLens
do
    # number of executors
    for executor_instances in 2
    do
        # param to enable bucketing
        for USE_BUCKETING in True
        do
            # Models to execution. List of available model presented in description of run_experiment.py
            for model in ALS_HNSWLIB ItemKNN_NMSLIB_HNSW SLIM_NMSLIB_HNSW
            do
                CORES_MAX=$(($EXECUTOR_CORES * $executor_instances))
                PARTITION_NUM=$((3 * $CORES_MAX))
                NUM_BLOCKS=$((3 * $CORES_MAX))
                EXPERIMENT="mlflow_experiment_name"

                # you can call submit.sh script directly in your shell if needed
                # i.e. without 'kubectl exec pod -- bash ...'
                kubectl -n spark-lama-exps exec spark-submit-3.2.0 -- \
                bash -c "export DATASET=$dataset \
                EXPERIMENT=$EXPERIMENT \
                SEED=$SEED \
                K=$K \
                K_LIST_METRICS=$K_LIST_METRICS \
                MODEL=$model \
                ALS_RANK=$ALS_RANK \
                NUM_NEIGHBOURS=$NUM_NEIGHBOURS \
                NUM_CLUSTERS=$NUM_CLUSTERS \
                WORD2VEC_RANK=$WORD2VEC_RANK \
                HNSWLIB_PARAMS=$HNSWLIB_PARAMS \
                NMSLIB_HNSW_PARAMS=$NMSLIB_HNSW_PARAMS \
                LOG_TO_MLFLOW=True \
                USE_BUCKETING=$USE_BUCKETING \
                USE_SCALA_UDFS_METRICS=$USE_SCALA_UDFS_METRICS \
                DATASETS_DIR=$DATASETS_DIR \
                RS_DATASETS_DIR=$RS_DATASETS_DIR \
                FORCE_RECREATE_DATASETS=$FORCE_RECREATE_DATASETS \
                CHECK_NUMBER_OF_ALLOCATED_EXECUTORS=$CHECK_NUMBER_OF_ALLOCATED_EXECUTORS \
                NUM_BLOCKS=$NUM_BLOCKS \
                PARTITION_NUM=$PARTITION_NUM \
                CORES_MAX=$CORES_MAX \
                EXECUTOR_MEMORY=$EXECUTOR_MEMORY \
                EXECUTOR_CORES=$EXECUTOR_CORES \
                DRIVER_MEMORY=$DRIVER_MEMORY \
                DRIVER_CORES=$DRIVER_CORES \
                DRIVER_MAX_RESULT_SIZE=$DRIVER_MAX_RESULT_SIZE \
                EXECUTOR_INSTANCES=$executor_instances \
                PYSPARK_PYTHON_PATH=$PYSPARK_PYTHON_PATH \
                SCRIPT=$SCRIPT \
                && submit.sh" \
                || continue
            done
        done
    done
done

# nohup ./bin/run_replay_experiments.sh > bin/experiment_logs/run_replay_experiments_$(date +%Y-%m-%d-%H%M%S).log  2>&1 &