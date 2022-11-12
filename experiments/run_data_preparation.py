import os
import logging.config

import mlflow
import pandas as pd
from pyspark.sql import SparkSession
from replay.data_preparator import DataPreparator, Indexer
from replay.session_handler import get_spark_session
from replay.utils import JobGroup, log_exec_timer

from rs_datasets import MovieLens, MillionSongDataset
from pyspark.sql import functions as sf
from replay.splitters import DateSplitter, UserSplitter
# from replay.utils import get_log_info2


VERBOSE_LOGGING_FORMAT = (
    "%(asctime)s %(levelname)s %(module)s %(filename)s:%(lineno)d %(message)s"
)
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger("replay")
logger.setLevel(logging.DEBUG)


def main(spark: SparkSession, dataset_name: str):
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://node2.bdcl:8811")
    # partition_num = int(
    #     spark.sparkContext.getConf().get("spark.cores.max")
    # )  # 28

    spark_conf = spark.sparkContext.getConf()

    if os.environ.get("PARTITION_NUM"):
        partition_num = int(os.environ.get("PARTITION_NUM"))
    else:
        if spark_conf.get("spark.cores.max") is None:
            partition_num = os.cpu_count()
        else:
            partition_num = int(spark_conf.get("spark.cores.max"))  # 28


    if dataset_name.startswith("MovieLens"):
        # name__size__sample pattern
        dataset_params = dataset_name.split("__")
        if len(dataset_params) == 1:
            dataset_version = "1m"
        elif len(dataset_params) == 2:
            dataset_version = dataset_params[1]
        elif len(dataset_params) == 3:
            dataset_version = dataset_params[1]
            n_samples = int(dataset_params[2])
        else:
            raise ValueError("Too many dataset params.")
        data = MovieLens(
            dataset_version, path="/opt/spark_data/replay_datasets/MovieLens"
        )
        data = data.ratings
        mapping = {
            "user_id": "user_id",
            "item_id": "item_id",
            "relevance": "rating",
            "timestamp": "timestamp",
        }
    elif dataset_name.startswith("MillionSongDataset"):
        # MillionSongDataset__{fraction} pattern
        dataset_params = dataset_name.split("__")
        if len(dataset_params) == 1:
            fraction = "1.0"
            data = MillionSongDataset(
                path="/opt/spark_data/replay_datasets/MillionSongDataset"
            )
            data = data.train
        else:
            fraction = dataset_params[1]
            if fraction == "train_10x_users":
                data = spark.read.parquet("file:///opt/spark_data/replay_datasets/MillionSongDataset/train_10x_users.parquet")
            else:
                data = pd.read_csv(f"/opt/spark_data/replay_datasets/MillionSongDataset/train_{fraction}.csv")

        # data = MillionSongDataset(
        #     path="/opt/spark_data/replay_datasets/MillionSongDataset"
        # )
        # data = pd.concat([data.train, data.test, data.val])

        mapping = {
            "user_id": "user_id",
            "item_id": "item_id",
            "relevance": "play_count",
        }
    # elif dataset_name == "ml1m":
    #     data = pd.read_csv("/opt/spark_data/replay_datasets/ml1m_ratings.dat", sep="\t", names=["userId", "item_id", "relevance", "timestamp"])
    #     mapping = {'user_id': 'userId',
    #                                   'item_id': 'item_id',
    #                                   'relevance': 'relevance',
    #                                   'timestamp': 'timestamp'
    #                                  }
    else:
        raise ValueError("Unknown dataset.")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(os.environ.get("EXPERIMENT", "Dataset_preparation")) # os.getenv("EXPERIMENT") f"replay-{MODEL}"

    with mlflow.start_run():

        mlflow.log_param(
            "spark.driver.cores",
            spark.sparkContext.getConf().get("spark.driver.cores"),
        )
        mlflow.log_param(
            "spark.driver.memory",
            spark.sparkContext.getConf().get("spark.driver.memory"),
        )
        mlflow.log_param(
            "spark.memory.fraction",
            spark.sparkContext.getConf().get("spark.memory.fraction"),
        )
        mlflow.log_param(
            "spark.executor.cores",
            spark.sparkContext.getConf().get("spark.executor.cores"),
        )
        mlflow.log_param(
            "spark.executor.memory",
            spark.sparkContext.getConf().get("spark.executor.memory"),
        )
        mlflow.log_param(
            "spark.executor.instances",
            spark.sparkContext.getConf().get("spark.executor.instances"),
        )
        mlflow.log_param(
            "spark.sql.shuffle.partitions",
            spark.sparkContext.getConf().get("spark.sql.shuffle.partitions"),
        )
        mlflow.log_param(
            "spark.default.parallelism",
            spark.sparkContext.getConf().get("spark.default.parallelism"),
        )
        mlflow.log_param(
            "spark.applicationId", spark.sparkContext.applicationId
        )

        mlflow.log_param("dataset", dataset_name)
        # mlflow.log_param("filter_log", os.getenv("FILTER_LOG") == "True")
        # mlflow.log_param("seed", SEED)
        # mlflow.log_param("K", K)

        with log_exec_timer("DataPreparator execution") as preparator_timer:
            preparator = DataPreparator()
            log = preparator.transform(columns_mapping=mapping, data=data)
            log = log.repartition(partition_num).cache()
            log.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("preparator_sec", preparator_timer.duration)

        mlflow.log_metric("log_num_partitions", log.rdd.getNumPartitions())

        if os.getenv("FILTER_LOG") == "True":
            with log_exec_timer("log filtering") as log_filtering_timer:
                # will consider ratings >= 3 as positive feedback. A positive feedback is treated with relevance = 1
                only_positives_log = log.filter(
                    sf.col("relevance") >= 1 # [3 or 1] try set 1
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

        with log_exec_timer("DateSplitter execution") as splitter_timer:
            if dataset_name.startswith("MovieLens"):
                # MovieLens
                train_spl = DateSplitter(
                    test_start=0.2,
                    drop_cold_items=True,
                    drop_cold_users=True,
                )
            else:
                ## MillionSongDataset
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

        
        if dataset_name.startswith("MillionSongDataset"):
            partition_nums = {6, 12, 24, 48}
            if partition_num in partition_nums:
                with log_exec_timer("Train/test datasets saving to parquet") as parquets_save_timer:
                    # WARN: 'fraction' is not fraction of test or train, it is fraction of input dataset.
                    train.write.mode('overwrite').parquet(f"/opt/spark_data/replay_datasets/MillionSongDataset/fraction_{fraction}_train_{partition_num}_partition.parquet")
                    test.write.mode('overwrite').parquet(f"/opt/spark_data/replay_datasets/MillionSongDataset/fraction_{fraction}_test_{partition_num}_partition.parquet")
                mlflow.log_metric(f"parquets{partition_num}_write_sec", parquets_save_timer.duration)
                partition_nums.remove(partition_num)
            for partition_num in partition_nums:
                with log_exec_timer("Train/test datasets saving to parquet") as parquets_save_timer:
                    train1 = train.repartition(partition_num)
                    test1 = test.repartition(partition_num)
                    # WARN: 'fraction' is not fraction of test or train, it is fraction of input dataset.
                    train1.write.mode('overwrite').parquet(f"/opt/spark_data/replay_datasets/MillionSongDataset/fraction_{fraction}_train_{partition_num}_partition.parquet")
                    test1.write.mode('overwrite').parquet(f"/opt/spark_data/replay_datasets/MillionSongDataset/fraction_{fraction}_test_{partition_num}_partition.parquet")
                mlflow.log_metric(f"parquets{partition_num}_write_sec", parquets_save_timer.duration)
        else:
            with log_exec_timer("Train/test datasets saving to parquet") as parquets_save_timer:
                train.write.mode('overwrite').parquet(f"/opt/spark_data/replay_datasets/MovieLens/train_{dataset_version}.parquet")
                test.write.mode('overwrite').parquet(f"/opt/spark_data/replay_datasets/MovieLens/test_{dataset_version}.parquet")
            mlflow.log_metric(f"parquets{partition_num}_write_sec", parquets_save_timer.duration)



if __name__ == "__main__":
    spark_sess = get_spark_session()
    dataset = os.getenv("DATASET")
    # dataset = "MovieLens__1m"
    # dataset = "MillionSongDataset"
    main(spark=spark_sess, dataset_name=dataset)
    spark_sess.stop()
