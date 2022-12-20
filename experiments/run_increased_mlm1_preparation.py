import logging.config
import os

import mlflow
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession

from replay.data_preparator import DataPreparator, Indexer
from replay.filters import filter_by_min_count, filter_out_low_ratings
from replay.session_handler import get_spark_session
from replay.splitters import DateSplitter
from replay.utils import get_log_info2
from replay.utils import log_exec_timer


VERBOSE_LOGGING_FORMAT = (
    "%(asctime)s %(levelname)s %(module)s %(filename)s:%(lineno)d %(message)s"
)
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger("replay")
logger.setLevel(logging.DEBUG)


def main(spark: SparkSession):
    K = int(os.environ.get("K", 5))
    SEED = int(os.environ.get("SEED", 1234))
    MLFLOW_TRACKING_URI = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8811"
    )

    spark_conf = spark.sparkContext.getConf()

    if os.environ.get("PARTITION_NUM"):
        partition_num = int(os.environ.get("PARTITION_NUM"))
    else:
        if spark_conf.get("spark.cores.max") is None:
            partition_num = os.cpu_count()
        else:
            partition_num = int(spark_conf.get("spark.cores.max"))  # 28

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("mlm1_preparation")

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
            "seed": SEED,
            "K": K
        }
        mlflow.log_params(spark_configs)

        # df = spark.read.parquet("file:///opt/spark_data/replay_datasets/MovieLens/train_ml1m_1m_users_3_7k_items.parquet")
        df = spark.read.parquet("file:///opt/spark_data/replay_datasets/MovieLens/train_ml1m_1m_users_37k_items.parquet")
        # df = pd.read_csv(
        #     "/opt/spark_data/replay_datasets/ml1m_ratings.dat",
        #     sep="\t",
        #     names=["userId", "item_id", "relevance", "timestamp"],
        # )
        # users = spark.read.parquet("file:///opt/spark_data/replay_datasets/MovieLens/train_ml1m_1m_users_3_7k_items_user_features.parquet")
        users = spark.read.parquet("file:///opt/spark_data/replay_datasets/MovieLens/train_ml1m_1m_users_37k_items_user_features.parquet")
        users = users.withColumn("age", users["age"].cast("int"))
        users = users.withColumn("occupation", users["occupation"].cast("int"))
        # users = pd.read_csv(
        #     "/opt/spark_data/replay_datasets/ml1m_users.dat",
        #     sep="\t",
        #     names=["user_id", "gender", "age", "occupation", "zip_code"],
        # )
        preparator = DataPreparator()
        log = preparator.transform(
            columns_mapping={
                "user_id": "user_id",
                "item_id": "item_id",
                "relevance": "rating",
                "timestamp": "timestamp",
            },
            data=df,
        )
        user_features = preparator.transform(
            columns_mapping={"user_id": "user_id"}, data=users
        )
        log = filter_out_low_ratings(log, value=3)
        log = filter_by_min_count(log, num_entries=5, group_by="user_id")
        indexer = Indexer(user_col="user_id", item_col="item_id")
        indexer.fit(
            users=log.select("user_id").unionByName(
                user_features.select("user_id")
            ),
            items=log.select("item_id"),
        )
        log_replay = indexer.transform(df=log)
        user_features = indexer.transform(df=user_features)
        # save_indexer(indexer, path='/tmp/ml1m_indexer', overwrite=True)
        # loaded_indexer = load_indexer('/tmp/ml1m_indexer')
        # tmp = loaded_indexer.transform(log)

        # Index StringType columns ("gender", "zip_code")
        user_feature_indexer = StringIndexer(
            inputCol="gender", outputCol="gender_idx"
        ).fit(user_features)
        user_features = user_feature_indexer.transform(user_features).select("user_idx", "gender_idx", "age", "occupation", "zip_code")

        user_feature_indexer = StringIndexer(
            inputCol="zip_code", outputCol="zip_code_idx"
        ).fit(user_features)
        user_features = user_feature_indexer.transform(user_features).select("user_idx", "gender_idx", "age", "occupation", "zip_code_idx")

        

        log_replay = log_replay.repartition(partition_num)
        log_replay = log_replay.cache()
        log_replay.write.mode("overwrite").format("noop").save()
        # splitter = UserSplitter(
        #     drop_cold_items=True,
        #     drop_cold_users=True,
        #     item_test_size=K,
        #     user_test_size=500,
        #     seed=SEED,
        #     shuffle=True,
        # )
        splitter = DateSplitter(
            test_start=0.2,
            drop_cold_items=True,
            drop_cold_users=True,

        )
        train, test = splitter.split(log_replay)
        # save_splitter(splitter, path='/tmp/ml1m_splitter', overwrite=True)
        # loaded_splitter = load_splitter('/tmp/ml1m_splitter')
        # tmp = loaded_splitter.split(log_replay)

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

        with log_exec_timer("Train/test/user_features datasets saving to parquet") as parquets_save_timer:
            # train.write.mode('overwrite').parquet(f"/opt/spark_data/replay_datasets/ml1m_1m_users_3_7k_items_train.parquet")
            # test.write.mode('overwrite').parquet(f"/opt/spark_data/replay_datasets/ml1m_1m_users_3_7k_items_test.parquet")
            # user_features.write.mode('overwrite').parquet(f"/opt/spark_data/replay_datasets/ml1m_1m_users_3_7k_items_user_features.parquet")
            train.write.mode('overwrite').parquet(f"/opt/spark_data/replay_datasets/ml1m_1m_users_37k_items_train.parquet")
            test.write.mode('overwrite').parquet(f"/opt/spark_data/replay_datasets/ml1m_1m_users_37k_items_test.parquet")
            user_features.write.mode('overwrite').parquet(f"/opt/spark_data/replay_datasets/ml1m_1m_users_37k_items_user_features.parquet")
            
        mlflow.log_metric(f"parquets{partition_num}_write_sec", parquets_save_timer.duration)



if __name__ == "__main__":
    spark_sess = get_spark_session()
    main(spark=spark_sess)
    spark_sess.stop()
