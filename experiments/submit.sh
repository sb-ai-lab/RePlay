#!/usr/bin/env bash

set -ex

SCRIPT="$1"

PYSPARK_PYTHON_PATH=${PYSPARK_PYTHON_PATH:-"/python_envs/.replay_venv/bin/python"}
DRIVER_CORES=${DRIVER_CORES:-"2"}
DRIVER_MEMORY=${DRIVER_MEMORY:-"20g"}
DRIVER_MAX_RESULT_SIZE=${DRIVER_MAX_RESULT_SIZE:-"5g"}
EXECUTOR_CORES=${EXECUTOR_CORES:-"6"}
EXECUTOR_MEMORY=${EXECUTOR_MEMORY:-"46g"}
FILTER_LOG=${FILTER_LOG:-"True"}
SEED=${SEED:-"42"}
K=${K:-"10"}
K_LIST_METRICS=${K_LIST_METRICS:-"5,10"}
USE_BUCKETING=${USE_BUCKETING:-"True"}
ALS_RANK=${ALS_RANK:-"100"}
NUM_NEIGHBOURS=${NUM_NEIGHBOURS:-"1000"}
NUM_CLUSTERS=${NUM_CLUSTERS:-"100"}
WORD2VEC_RANK=${WORD2VEC_RANK:-"100"}
HNSWLIB_PARAMS=${HNSWLIB_PARAMS:-'{\"space\":\"ip\",\"M\":100,\"efS\":2000,\"efC\":2000,\"post\":0,\"index_path\":\"/tmp/hnswlib_index_{spark_app_id}\",\"build_index_on\":\"executor\"}'}
NMSLIB_HNSW_PARAMS=${NMSLIB_HNSW_PARAMS:-'{\"method\":\"hnsw\",\"space\":\"negdotprod_sparse_fast\",\"M\":100,\"efS\":2000,\"efC\":2000,\"post\":0,\"index_path\":\"/tmp/nmslib_hnsw_index_{spark_app_id}\",\"build_index_on\":\"executor\"}'}

WAREHOUSE_DIR=${WAREHOUSE_DIR:-"hdfs://node21.bdcl:9000/spark-warehouse"}
DATASETS_DIR=${DATASETS_DIR:-"/opt/spark_data/replay_datasets/"}
RS_DATASETS_DIR=${RS_DATASETS_DIR:-"/opt/spark_data/replay_datasets/"}
FORCE_RECREATE_DATASETS=${FORCE_RECREATE_DATASETS:-"False"}
CHECK_NUMBER_OF_ALLOCATED_EXECUTORS=${CHECK_NUMBER_OF_ALLOCATED_EXECUTORS:-"True"}

EXECUTOR_INSTANCES=${EXECUTOR_INSTANCES:-"2"}
DATASET=${DATASET:-"MovieLens"}
MODEL=${MODEL:-"ALS_HNSWLIB"}
EXPERIMENT=$MODEL"_$(cut -d'_' -f1 <<<$DATASET)"

# calculable variables
# shellcheck disable=SC2004
CORES_MAX=$(($EXECUTOR_CORES * $EXECUTOR_INSTANCES))
# shellcheck disable=SC2004
PARTITION_NUM=$((3 * $CORES_MAX))
NUM_BLOCKS=$((3 * $CORES_MAX))



spark-submit \
--master yarn \
--deploy-mode cluster \
--conf 'spark.yarn.appMasterEnv.SCRIPT_ENV=cluster' \
--conf 'spark.yarn.appMasterEnv.PYSPARK_PYTHON='${PYSPARK_PYTHON_PATH} \
--conf 'spark.yarn.appMasterEnv.MLFLOW_TRACKING_URI=http://node2.bdcl:8822' \
--conf 'spark.yarn.appMasterEnv.DATASET='${DATASET} \
--conf 'spark.yarn.appMasterEnv.SEED='${SEED} \
--conf 'spark.yarn.appMasterEnv.K='${K} \
--conf 'spark.yarn.appMasterEnv.K_LIST_METRICS='${K_LIST_METRICS} \
--conf 'spark.yarn.appMasterEnv.MODEL='${MODEL} \
--conf 'spark.yarn.appMasterEnv.ALS_RANK='${ALS_RANK} \
--conf 'spark.yarn.appMasterEnv.NUM_NEIGHBOURS='${NUM_NEIGHBOURS} \
--conf 'spark.yarn.appMasterEnv.NUM_CLUSTERS='${NUM_CLUSTERS} \
--conf 'spark.yarn.appMasterEnv.WORD2VEC_RANK='${WORD2VEC_RANK} \
--conf 'spark.yarn.appMasterEnv.HNSWLIB_PARAMS='${HNSWLIB_PARAMS} \
--conf 'spark.yarn.appMasterEnv.NMSLIB_HNSW_PARAMS='${NMSLIB_HNSW_PARAMS} \
--conf 'spark.yarn.appMasterEnv.USE_RELEVANCE='${USE_RELEVANCE} \
--conf 'spark.yarn.appMasterEnv.LOG_TO_MLFLOW='${LOG_TO_MLFLOW} \
--conf 'spark.yarn.appMasterEnv.USE_SCALA_UDFS_METRICS='${USE_SCALA_UDFS_METRICS} \
--conf 'spark.yarn.appMasterEnv.EXPERIMENT='${EXPERIMENT} \
--conf 'spark.yarn.appMasterEnv.FILTER_LOG='${FILTER_LOG} \
--conf 'spark.yarn.appMasterEnv.NUM_BLOCKS='${NUM_BLOCKS} \
--conf 'spark.yarn.appMasterEnv.PARTITION_NUM='${PARTITION_NUM} \
--conf 'spark.yarn.appMasterEnv.USE_BUCKETING='${USE_BUCKETING} \
--conf 'spark.yarn.appMasterEnv.DATASETS_DIR='${DATASETS_DIR} \
--conf 'spark.yarn.appMasterEnv.RS_DATASETS_DIR='${RS_DATASETS_DIR} \
--conf 'spark.yarn.appMasterEnv.FORCE_RECREATE_DATASETS='${FORCE_RECREATE_DATASETS} \
--conf 'spark.yarn.appMasterEnv.CHECK_NUMBER_OF_ALLOCATED_EXECUTORS='${CHECK_NUMBER_OF_ALLOCATED_EXECUTORS} \
--conf 'spark.yarn.appMasterEnv.GIT_PYTHON_REFRESH=quiet' \
--conf "spark.yarn.tags=replay" \
--conf 'spark.kryoserializer.buffer.max=2010m' \
--conf 'spark.driver.cores='${DRIVER_CORES} \
--conf 'spark.driver.memory='${DRIVER_MEMORY} \
--conf 'spark.driver.maxResultSize='${DRIVER_MAX_RESULT_SIZE} \
--conf 'spark.executor.instances='${EXECUTOR_INSTANCES} \
--conf 'spark.executor.cores='${EXECUTOR_CORES} \
--conf 'spark.executor.memory='${EXECUTOR_MEMORY} \
--conf 'spark.cores.max='${CORES_MAX} \
--conf 'spark.memory.fraction=0.4' \
--conf 'spark.sql.shuffle.partitions='${PARTITION_NUM} \
--conf 'spark.default.parallelism='${PARTITION_NUM} \
--conf 'spark.yarn.maxAppAttempts=1' \
--conf 'spark.rpc.message.maxSize=1024' \
--conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
--conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
--conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
--conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
--conf 'spark.eventLog.enabled=true' \
--conf 'spark.eventLog.dir=hdfs://node21.bdcl:9000/shared/spark-logs' \
--conf 'spark.yarn.historyServer.allowTracking=true' \
--conf 'spark.driver.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true' \
--conf 'spark.executor.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true' \
--conf 'spark.executor.extraClassPath=/jars/replay_jars/*' \
--conf 'spark.driver.extraClassPath=/jars/replay_jars/*' \
--conf 'spark.sql.warehouse.dir='${WAREHOUSE_DIR} \
--conf 'spark.task.maxFailures=1' \
--conf 'spark.excludeOnFailure.task.maxTaskAttemptsPerNode=1' \
--conf 'spark.excludeOnFailure.stage.maxFailedTasksPerExecutor=1' \
--conf 'spark.excludeOnFailure.stage.maxFailedExecutorsPerNode=1' \
--conf 'spark.excludeOnFailure.application.maxFailedTasksPerExecutor=1' \
--conf 'spark.excludeOnFailure.application.maxFailedExecutorsPerNode=1' \
--conf 'spark.python.worker.reuse=true' \
--conf 'spark.sql.optimizer.maxIterations=100' \
--conf 'spark.files.overwrite=true' \
--py-files 'dist/replay_rec-0.10.0-py3-none-any.whl,experiments/experiment_utils.py' \
--num-executors ${EXECUTOR_INSTANCES} \
--jars 'scala/target/scala-2.12/replay_2.12-0.1.jar' \
"${SCRIPT}"

# launch example:
# ./experiments/submit.sh experiments/run_experiment.py
