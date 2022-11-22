import copy
import os
from typing import Optional, Tuple

import pyspark.sql.functions as sf

# import numpy as np
# import pandas as pd
import mlflow
# import nmslib
# import tempfile

# from pyarrow import fs

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.functions import array_to_vector
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import DoubleType

from replay.models.base_rec import Recommender, ItemVectorModel
from replay.models.nmslib_hnsw import NmslibHnswMixin
from replay.utils import JobGroup, list_to_vector_udf, log_exec_timer
# from replay.utils import get_top_k_recs


class ALSWrap(Recommender, ItemVectorModel, NmslibHnswMixin):
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
        nmslib_hnsw_params: Optional[dict] = None,
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
        self._nmslib_hnsw_params = nmslib_hnsw_params

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

        if self._nmslib_hnsw_params:
            item_vectors, _ = self.get_features(
                log.select("item_idx").distinct()
            )

            self._build_hnsw_index(item_vectors, 'item_factors', self._nmslib_hnsw_params)

            self._user_to_max_items = (
                    log.groupBy('user_idx')
                    .agg(sf.count('item_idx').alias('num_items'))
            )

    def _clear_cache(self):
        if hasattr(self, "model"):
            self.model.itemFactors.unpersist()
            self.model.userFactors.unpersist()

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
        
        if self._nmslib_hnsw_params:

            params = self._nmslib_hnsw_params

            with JobGroup(
                f"{self.__class__.__name__}.get_features()",
                "Model inference (inside 1.5)",
            ):
                user_vectors, _ = self.get_features(users)
                # user_vectors = user_vectors.cache()
                # user_vectors.write.mode("overwrite").format("noop").save()

                user_vectors = user_vectors.join(self._user_to_max_items, on="user_idx")

            res = self._infer_hnsw_index(user_vectors, "user_factors", params, k)

            return res

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

            with JobGroup(
                f"{self.__class__.__name__}.model.recommendForUserSubset()",
                "Model inference (inside 1.4)",
            ):
                recs_als = self.model.recommendForUserSubset(
                    users, k + max_seen
                )
                recs_als = recs_als.cache()
                recs_als.write.mode("overwrite").format("noop").save()

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
