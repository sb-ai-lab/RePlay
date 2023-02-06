import os
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import pyspark.sql.functions as sf
import mlflow
import numpy as np

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.linear_model import Ridge

from replay.models.base_rec import Recommender, ItemVectorModel
from replay.models.hnswlib import HnswlibMixin
from replay.session_handler import State
from replay.spark_custom_models.recommendation import ALS, ALSModel
from replay.utils import list_to_vector_udf


class ALSWrap(Recommender, ItemVectorModel, HnswlibMixin):
    """Wrapper for `Spark ALS
    <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS>`_.
    """

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        return {
            "features_col": "user_factors",
            "params": self._hnswlib_params,
            "index_dim": self.rank,
        }

    def _get_vectors_to_infer_ann_inner(self, log: DataFrame, users: DataFrame) -> DataFrame:
        user_vectors, _ = self.get_features(users)
        return user_vectors

    def _get_ann_build_params(self, log: DataFrame):
        self.num_elements = log.select("item_idx").distinct().count()
        return {
            "features_col": "item_factors",
            "params": self._hnswlib_params,
            "dim": self.rank,
            "num_elements": self.num_elements,
            "id_col": "item_idx",
        }

    def _get_vectors_to_build_ann(self, log: DataFrame) -> DataFrame:
        item_vectors, _ = self.get_features(
            log.select("item_idx").distinct()
        )
        return item_vectors

    @property
    def _use_ann(self) -> bool:
        return self._hnswlib_params is not None

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
        hnswlib_params: Optional[dict] = None,
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
        self._hnswlib_params = hnswlib_params

    @property
    def _init_args(self):
        return {
            "rank": self.rank,
            "implicit_prefs": self.implicit_prefs,
            "seed": self._seed,
            "hnswlib_params": self._hnswlib_params
        }

    def _save_model(self, path: str):
        self.model.write().overwrite().save(path)

        if self._hnswlib_params:
            self._save_hnswlib_index(path)

    def _load_model(self, path: str):
        self.model = ALSModel.load(path)
        self.model.itemFactors.cache()
        self.model.userFactors.cache()

        if self._hnswlib_params:
            self._load_hnswlib_index(path)

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
        self.model.itemFactors.cache()
        self.model.userFactors.cache()
        self.model.itemFactors.count()
        self.model.userFactors.count()

    def fit_partial(self, log: DataFrame, previous_log: Optional[Union[str, DataFrame]] = None, merged_log_path: Optional[str] = None) -> None:
        new_users = log.select('user_idx').distinct().join(previous_log.select('user_idx').distinct(), how='left_anti', on='user_idx')
        old_items = previous_log.select('item_idx').distinct()
        old_users = previous_log.select('user_idx').distinct()
        new_items = log.select('item_idx').distinct().join(old_items, how='left_anti', on='item_idx')

        train_union = previous_log.union(log)

        train_union_new_users_old_items = train_union.join(new_users, how='inner', on='user_idx').join(old_items, how='inner', on='item_idx')

        items_count = train_union.agg({"item_idx": "max"}).first()[0] + 1
        users_count = train_union.agg({"user_idx": "max"}).first()[0] + 1

        train_union_new_users_old_items = train_union_new_users_old_items.toPandas()

        interactions_matrix_new_users_old_items = csr_matrix(
            (train_union_new_users_old_items.relevance, (train_union_new_users_old_items.user_idx, train_union_new_users_old_items.item_idx)),
            shape=(users_count, items_count))

        interactions_matrix_broadcast = (
                State().session.sparkContext.broadcast(interactions_matrix_new_users_old_items)
        )

        df_item_factors = self.model.itemFactors.toPandas()
        X_regr = np.zeros((items_count, self.rank))
        for i, row in df_item_factors.iterrows():
            X_regr[int(row['id'])] = row['features']
            
        old_items_values = old_items.toPandas()['item_idx'].values
        X_regr = X_regr[old_items_values]
        X_regr_broadcast = (
                State().session.sparkContext.broadcast(X_regr)
        )

        SOLVER = 'auto' #'lsqr' 'sag'
        MAX_ITER = None
        reduction = 10

        rank = self.rank

        @udf(returnType=ArrayType(FloatType(), True)) 
        def compute_new_user_factors_udf(user_id: int):
            interactions_matrix = interactions_matrix_broadcast.value
            Y_regr = interactions_matrix[user_id].toarray()[0][old_items_values]
            non_zero_indexes = np.nonzero(Y_regr)[0]
            zero_indexes = np.random.choice(np.where(Y_regr == 0)[0], int((Y_regr.shape[0]-non_zero_indexes.shape[0])/reduction))  # what if count(1) > count(0) ?
            usefull_indexes = np.concatenate([non_zero_indexes, zero_indexes])
            if usefull_indexes.shape[0] == 0:
                return [float(x) for x in np.zeros(rank)]

            reg_model = Ridge(alpha=.75, solver=SOLVER, max_iter=MAX_ITER)

            X_regr = X_regr_broadcast.value
            reg_model.fit(X_regr[usefull_indexes], Y_regr[usefull_indexes])

            return [float(x) for x in reg_model.coef_]

        # getting factors for new users
        df_new_user_factors = new_users.select(sf.col("user_idx").alias("id"), compute_new_user_factors_udf("user_idx").alias("features"))

        # unioning old and new user factors
        self.df_user_factors_total = self.model.userFactors.union(df_new_user_factors)

        # new items
        train_union_all_users_new_items = train_union.join(new_items, how='inner', on='item_idx').join(old_users, how='inner', on='user_idx') #old users + new items

        train_union_all_users_new_items = train_union_all_users_new_items.toPandas()

        interactions_matrix_all_users_new_items =  csc_matrix(
            (train_union_all_users_new_items.relevance, (train_union_all_users_new_items.user_idx, train_union_all_users_new_items.item_idx)),
            shape=(users_count, items_count))

        interactions_matrix_broadcast = (
                State().session.sparkContext.broadcast(interactions_matrix_all_users_new_items)
        )

        all_users = train_union_all_users_new_items['user_idx'].unique()

        X_regr = np.zeros((users_count, self.rank))
        for i, row in self.df_user_factors_total.toPandas().iterrows(): #df_user_factors
            X_regr[int(row['id'])] = row['features'] # a lot of zero-features users 
            
        # new_items_values = new_items.toPandas()['item_idx'].values
        X_regr = X_regr[all_users]
        X_regr_broadcast = (
                State().session.sparkContext.broadcast(X_regr)
        )

        @udf(returnType=ArrayType(FloatType(), True))
        def compute_new_item_factors(item_id: int):
            interactions_matrix = interactions_matrix_broadcast.value
            Y_regr = interactions_matrix[:, item_id].toarray()[all_users]
            non_zero_indexes = np.nonzero(Y_regr)[0]
            try:
                # zero_indexes = np.random.choice(np.where(Y_regr == 0)[0], non_zero_indexes.shape[0])  # what if count(1) > count(0) ?
                zero_indexes = np.random.choice(np.where(Y_regr == 0)[0], int((Y_regr.shape[0]-non_zero_indexes.shape[0])/reduction))  # what if count(1) > count(0) ?
            except:
                return [float(x) for x in np.zeros(rank)]
            usefull_indexes = np.concatenate([non_zero_indexes, zero_indexes])
            if usefull_indexes.shape[0] == 0:
                return [float(x) for x in np.zeros(rank)]

            X_regr = X_regr_broadcast.value
            reg_model = Ridge(alpha=.75, solver=SOLVER, max_iter=MAX_ITER)
            reg_model.fit(X_regr[usefull_indexes], Y_regr[usefull_indexes].flatten())
            
            return [float(x) for x in reg_model.coef_]

        # getting factors for new items
        df_new_item_factors = new_items.select(sf.col("item_idx").alias("id"), compute_new_item_factors("item_idx").alias("features"))

        # unioning old and new item factors
        # self.df_item_factors_total = self.model.itemFactors.union(df_new_item_factors)

        self.num_elements = self.num_elements + df_new_item_factors.count()
        print(f"new index 'num_elements' = {self.num_elements}")
        self._update_hnsw_index(df_new_item_factors, 'features', self._hnswlib_params, self.rank, self.num_elements)

        self._user_to_max_items = (
                train_union.groupBy('user_idx')
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

        recs_als = self.model.recommendItemsForUserItemSubset(
            users, items, k + max_seen
        )
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
