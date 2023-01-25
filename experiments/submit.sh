#!/usr/bin/env bash

set -ex

spark-submit \
--master yarn \
--deploy-mode cluster \
--conf 'spark.yarn.appMasterEnv.SCRIPT_ENV=cluster' \
--conf 'spark.yarn.appMasterEnv.PYSPARK_PYTHON='${PYSPARK_PYTHON_PATH} \
--conf 'spark.yarn.appMasterEnv.MLFLOW_TRACKING_URI=http://node2.bdcl:8822' \
--conf 'spark.yarn.appMasterEnv.DATASET='${DATASET} \
--conf 'spark.yarn.appMasterEnv.SEED='${SEED} \
--conf 'spark.yarn.appMasterEnv.K='${K} \
--conf 'spark.yarn.appMasterEnv.MODEL='${MODEL} \
--conf 'spark.yarn.appMasterEnv.ALS_RANK='${ALS_RANK} \
--conf 'spark.yarn.appMasterEnv.NUM_NEIGHBOURS='${NUM_NEIGHBOURS} \
--conf 'spark.yarn.appMasterEnv.NUM_CLUSTERS='${NUM_CLUSTERS} \
--conf 'spark.yarn.appMasterEnv.WORD2VEC_RANK='${WORD2VEC_RANK} \
--conf 'spark.yarn.appMasterEnv.USE_RELEVANCE='${USE_RELEVANCE} \
--conf 'spark.yarn.appMasterEnv.LOG_TO_MLFLOW='${LOG_TO_MLFLOW} \
--conf 'spark.yarn.appMasterEnv.USE_SCALA_UDFS_METRICS='${USE_SCALA_UDFS_METRICS} \
--conf 'spark.yarn.appMasterEnv.EXPERIMENT='${EXPERIMENT} \
--conf 'spark.yarn.appMasterEnv.FILTER_LOG='${FILTER_LOG} \
--conf 'spark.yarn.appMasterEnv.NUM_BLOCKS='${NUM_BLOCKS} \
--conf 'spark.yarn.appMasterEnv.PARTITION_NUM='${PARTITION_NUM} \
--conf 'spark.yarn.appMasterEnv.USE_BUCKETING='${USE_BUCKETING} \
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
--conf 'spark.sql.warehouse.dir=hdfs://node21.bdcl:9000/spark-warehouse' \
--conf 'spark.task.maxFailures=1' \
--conf 'spark.excludeOnFailure.task.maxTaskAttemptsPerNode=1' \
--conf 'spark.excludeOnFailure.stage.maxFailedTasksPerExecutor=1' \
--conf 'spark.excludeOnFailure.stage.maxFailedExecutorsPerNode=1' \
--conf 'spark.excludeOnFailure.application.maxFailedTasksPerExecutor=1' \
--conf 'spark.excludeOnFailure.application.maxFailedExecutorsPerNode=1' \
--conf 'spark.python.worker.reuse=true' \
--conf 'spark.sql.optimizer.maxIterations=100' \
--conf 'spark.files.overwrite=true' \
--py-files '/submit_files/replay/replay_rec-0.10.0-py3-none-any.whl,/submit_files/replay/experiment_utils.py' \
--num-executors ${EXECUTOR_INSTANCES} \
--jars '/submit_files/replay/replay_2.12-0.1.jar' \
$SCRIPT

