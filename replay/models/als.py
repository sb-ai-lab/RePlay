import os
from typing import Optional, Tuple

import pyspark.sql.functions as sf

import mlflow
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.functions import array_to_vector
from pyspark_hnsw.knn import HnswSimilarity
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import DoubleType


from replay.models.base_rec import Recommender, ItemVectorModel
from replay.utils import JobGroup, list_to_vector_udf, log_exec_timer
from replay.utils import get_top_k_recs


class ALSWrap(Recommender, ItemVectorModel):
    """Wrapper for `Spark ALS
    <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS>`_.
    """

    _seed: Optional[int] = None
    _search_space = {
        "rank": {"type": "loguniform_int", "args": [8, 256]},
    }

    def __init__(
        self,
        rank: int = 10,
        implicit_prefs: bool = True,
        seed: Optional[int] = None,
        num_item_blocks: Optional[int] = None,
        num_user_blocks: Optional[int] = None,
        hnsw_params: Optional[dict] = None,
    ):
        """
        :param rank: hidden dimension for the approximate matrix
        :param implicit_prefs: flag to use implicit feedback
        :param seed: random seed
        """
        self.rank = rank
        self.implicit_prefs = implicit_prefs
        self._seed = seed
        self._num_item_blocks = num_item_blocks
        self._num_user_blocks = num_user_blocks
        self._hnsw_params = hnsw_params

    @property
    def _init_args(self):
        return {
            "rank": self.rank,
            "implicit_prefs": self.implicit_prefs,
            "seed": self._seed,
        }

    def _save_model(self, path: str):
        self.model.write().overwrite().save(path)

    def _load_model(self, path: str):
        self.model = ALSModel.load(path)
        self.model.itemFactors.cache()
        self.model.userFactors.cache()

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        if self._num_item_blocks is None:
            self._num_item_blocks = log.rdd.getNumPartitions()
        if self._num_user_blocks is None:
            self._num_user_blocks = log.rdd.getNumPartitions()

        with log_exec_timer("ALS.fit() execution") as als_fit_timer:
            self.model = ALS(
                rank=self.rank,
                numItemBlocks=self._num_item_blocks,
                numUserBlocks=self._num_user_blocks,
                userCol="user_idx",
                itemCol="item_idx",
                ratingCol="relevance",
                implicitPrefs=self.implicit_prefs,
                seed=self._seed,
                coldStartStrategy="drop",
            ).fit(log)
        if os.environ.get("LOG_TO_MLFLOW", None) == "True":
            mlflow.log_param("num_blocks", self._num_item_blocks)
            mlflow.log_metric("als_fit_sec", als_fit_timer.duration)
        self.model.itemFactors.cache()
        self.model.userFactors.cache()
        self.model.itemFactors.count()
        self.model.userFactors.count()

        if self._hnsw_params:
            item_vectors, _ = self.get_features(
                log.select("item_idx").distinct()
            )

            max_items_to_retrieve, *_ = (
                log.groupBy("user_idx")
                .agg(sf.count("item_idx").alias("num_items"))
                .select(sf.max("num_items"))
                .first()
            )

            hnsw = HnswSimilarity(
                identifierCol="id",
                featuresCol="features",
                queryIdentifierCol="user_id",
                distanceFunction=self._hnsw_params["distanceFunction"],
                m=self._hnsw_params["m"],
                ef=self._hnsw_params["ef"],
                k=self._hnsw_params["k"] + max_items_to_retrieve,
                efConstruction=self._hnsw_params["efConstruction"],
                numPartitions=self._hnsw_params["numPartitions"],  # log.rdd.getNumPartitions()
                excludeSelf=self._hnsw_params["excludeSelf"],
            )

            to_index = item_vectors.select(
                sf.col("item_idx").alias("id"),
                array_to_vector("item_factors").alias("features"),
            )

            self._hnsw_model = hnsw.fit(to_index)

            # self._hnsw_model.write().overwrite().save("/tmp/hnsw_model")

    def _clear_cache(self):
        if hasattr(self, "model"):
            self.model.itemFactors.unpersist()
            self.model.userFactors.unpersist()

    def _get_executors_number_and_cores_per_executor(self) -> int:
        spark_conf = SparkSession.getActiveSession().sparkContext.getConf()
        master_addr = spark_conf.get("spark.master")
        if master_addr.startswith("local"):
            executors = 1

            # https://spark.apache.org/docs/latest/submitting-applications.html#master-urls
            # formats: local, local[K], local[K,F], local[*], local[*,F]
            if master_addr == "local":
                cores = 1
            else:
                cores_str = master_addr[len("local[") : -1]
                cores_str = cores_str.split(",")[0]
                cores = int(cores_str) if cores_str != "*" else os.cpu_count()
        else:
            executors = int(spark_conf.get("spark.executor.instances"))

            cores = int(spark_conf.get("spark.executor.cores"))

        return executors, cores

    def _get_neighbours_pyspark_hnsw(self, searcher, user_vectors_df):
        partition_num = user_vectors_df.rdd.getNumPartitions()
        executors, cores = self._get_executors_number_and_cores_per_executor()

        # configuration to improve performance and reduce shuffle
        user_vectors_df = user_vectors_df.repartition(int(executors))
        searcher.setParallelism(int(cores))

        with JobGroup(
            f"{searcher.__class__.__name__}.transform()",
            "Model inference (inside 1.2.1)",
        ):
            neighbours = searcher.transform(user_vectors_df)
            neighbours = neighbours.cache()
            neighbours.write.mode("overwrite").format("noop").save()

        neighbours = neighbours.repartition(partition_num)

        neighbours = neighbours.select(
            neighbours.user_id, sf.explode(neighbours.prediction)
        )
        neighbours = (
            neighbours.withColumn("item_id", sf.col("col.neighbor"))
            .withColumn("distance", sf.col("col.distance"))
            .withColumn("relevance", sf.lit(-1.0) * sf.col("distance"))
        )
        neighbours = neighbours.select("user_id", "item_id", "relevance")

        spark_res = neighbours
        spark_res = spark_res.withColumnRenamed(
            "item_id", "item_idx"
        ).withColumnRenamed("user_id", "user_idx")

        return spark_res

    def _filter_seen_hnsw_res(self, log, pred, k, id_type="idx"):
        """
        filter items seen in log and leave top-k most relevant
        """

        user_id = "user_" + id_type
        item_id = "item_" + id_type
        num_of_seen = log.groupBy(user_id).agg(
            sf.count(item_id).alias("seen_count")
        )

        max_seen = num_of_seen.select(sf.max("seen_count")).collect()[0][0]

        recs = pred.withColumn(
            "temp_rank",
            sf.row_number().over(
                Window.partitionBy(user_id).orderBy(sf.col("relevance").desc())
            ),
        ).filter(sf.col("temp_rank") <= sf.lit(max_seen + k))

        recs = (
            recs.join(num_of_seen, on=user_id, how="left")
            .fillna(0)
            .filter(sf.col("temp_rank") <= sf.col("seen_count") + sf.lit(k))
            .drop("temp_rank", "seen_count")
        )

        recs = recs.join(log, on=[user_id, item_id], how="anti")
        return get_top_k_recs(recs, k, id_type=id_type)

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: Optional[DataFrame],
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        if self._hnsw_params:
            with JobGroup(
                f"{self.__class__.__name__}.get_features()",
                "Model inference (inside 1.1)",
            ):
                user_vectors, _ = self.get_features(users)
                user_vectors = user_vectors.cache()
                user_vectors.write.mode("overwrite").format("noop").save()

            with JobGroup(
                f"select(... array_to_vector())",
                "Model inference (inside 1.1.1)",
            ):
                user_vectors_df = user_vectors.select(
                    sf.col("user_idx").alias("user_id"),
                    array_to_vector("user_factors").alias("features"),
                )
                user_vectors_df = user_vectors_df.cache()
                user_vectors_df.write.mode("overwrite").format("noop").save()

            with JobGroup(
                f"{self.__class__.__name__}._get_neighbours_pyspark_hnsw()",
                "Model inference (inside 1.2)",
            ):
                hnsw_res = self._get_neighbours_pyspark_hnsw(
                    self._hnsw_model, user_vectors_df
                )
                hnsw_res = hnsw_res.cache()
                hnsw_res.write.mode("overwrite").format("noop").save()

            with JobGroup(
                f"{self.__class__.__name__}._filter_seen2()",
                "Model inference (inside 1.3)",
            ):
                hnsw_res = self._filter_seen_hnsw_res(log, hnsw_res, k)
                hnsw_res.write.mode("overwrite").format("noop").save()
            hnsw_res = hnsw_res.cache()

            return hnsw_res

        if (items.count() == self.fit_items.count()) and (
            items.join(self.fit_items, on="item_idx", how="inner").count()
            == self.fit_items.count()
        ):
            max_seen = 0
            if filter_seen_items and log is not None:
                max_seen_in_log = (
                    log.join(users, on="user_idx")
                    .groupBy("user_idx")
                    .agg(sf.count("user_idx").alias("num_seen"))
                    .select(sf.max("num_seen"))
                    .collect()[0][0]
                )
                max_seen = (
                    max_seen_in_log if max_seen_in_log is not None else 0
                )

            recs_als = self.model.recommendForUserSubset(users, k + max_seen)
            mlflow.log_metric("als_predict_branch", 1)
            return (
                recs_als.withColumn(
                    "recommendations", sf.explode("recommendations")
                )
                .withColumn("item_idx", sf.col("recommendations.item_idx"))
                .withColumn(
                    "relevance",
                    sf.col("recommendations.rating").cast(DoubleType()),
                )
                .select("user_idx", "item_idx", "relevance")
            )

        mlflow.log_metric("als_predict_branch", 2)
        return self._predict_pairs(
            pairs=users.crossJoin(items).withColumn("relevance", sf.lit(1)),
            log=log,
        )

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        return (
            self.model.transform(pairs)
            .withColumn("relevance", sf.col("prediction").cast(DoubleType()))
            .drop("prediction")
        )

    def _get_features(
        self, ids: DataFrame, features: Optional[DataFrame]
    ) -> Tuple[Optional[DataFrame], Optional[int]]:
        entity = "user" if "user_idx" in ids.columns else "item"
        als_factors = getattr(self.model, f"{entity}Factors")
        als_factors = als_factors.withColumnRenamed(
            "id", f"{entity}_idx"
        ).withColumnRenamed("features", f"{entity}_factors")
        return (
            als_factors.join(ids, how="right", on=f"{entity}_idx"),
            self.model.rank,
        )

    def _get_item_vectors(self):
        return self.model.itemFactors.select(
            sf.col("id").alias("item_idx"),
            list_to_vector_udf(sf.col("features")).alias("item_vector"),
        )
