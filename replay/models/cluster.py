from typing import Optional, Union

from pyspark.sql import DataFrame
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as sf

from replay.models.base_rec import UserRecommender
from replay.models.nmslib_hnsw import NmslibHnswMixin


class ClusterRec(UserRecommender, NmslibHnswMixin):
    """
    Generate recommendations for cold users using k-means clusters
    """

    can_predict_cold_users = True
    _search_space = {
        "num_clusters": {"type": "int", "args": [2, 20]},
    }
    item_rel_in_cluster: DataFrame

    def __init__(
        self, num_clusters: int = 10, nmslib_hnsw_params: Optional[dict] = None
    ):
        """
        :param num_clusters: number of clusters
        """
        self.num_clusters = num_clusters
        self._nmslib_hnsw_params = nmslib_hnsw_params

    @property
    def _init_args(self):
        return {"num_clusters": self.num_clusters}

    def _save_model(self, path: str):
        self.model.write().overwrite().save(path)

    def _load_model(self, path: str):
        self.model = KMeansModel.load(path)

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        kmeans = KMeans().setK(self.num_clusters).setFeaturesCol("features")
        user_features_vector = self._transform_features(user_features)
        self.model = kmeans.fit(user_features_vector)
        self.users_clusters = (
            self.model.transform(user_features_vector)
            .select("user_idx", "prediction")
            .withColumnRenamed("prediction", "cluster")
        )

        log = log.join(self.users_clusters, on="user_idx", how="left")
        self.item_count_in_cluster = log.groupBy(["cluster", "item_idx"]).agg(
            sf.count("item_idx").alias("item_count")
        )

        max_count_per_cluster = self.item_count_in_cluster.groupby(
            "cluster"
        ).agg(sf.max("item_count").alias("max_count_in_cluster"))
        self.item_rel_in_cluster = self.item_count_in_cluster.join(
            max_count_per_cluster, on="cluster"
        )
        self.item_rel_in_cluster = self.item_rel_in_cluster.withColumn(
            "relevance", sf.col("item_count") / sf.col("max_count_in_cluster")
        ).drop("item_count", "max_count_in_cluster")
        self.item_rel_in_cluster.cache().count()

        # if self._nmslib_hnsw_params:
        #     item_vectors = log.select("item_idx", "cluster").distinct()

        #     self._build_hnsw_index(item_vectors, features_col='cluster', params=self._nmslib_hnsw_params)

        #     self._user_to_max_items = (
        #             log.groupBy('user_idx')
        #             .agg(sf.count('item_idx').alias('num_items'))
        #     )

    def refit(
        self,
        log: DataFrame,
        previous_log: Optional[Union[str, DataFrame]] = None,
        merged_log_path: Optional[str] = None,
    ) -> None:

        log = log.join(self.users_clusters, on="user_idx", how="left")
        item_count_in_cluster = log.groupBy(["cluster", "item_idx"]).agg(
            sf.count("item_idx").alias("item_count")
        )

        self.item_count_in_cluster = (
            self.item_count_in_cluster.union(item_count_in_cluster)
            .groupBy(["cluster", "item_idx"])
            .agg(
                sf.sum("item_count").alias("item_count"),
            )
        )

        max_count_per_cluster = self.item_count_in_cluster.groupby(
            "cluster"
        ).agg(sf.max("item_count").alias("max_count_in_cluster"))
        self.item_rel_in_cluster = self.item_count_in_cluster.join(
            max_count_per_cluster, on="cluster"
        )
        self.item_rel_in_cluster = self.item_rel_in_cluster.withColumn(
            "relevance", sf.col("item_count") / sf.col("max_count_in_cluster")
        ).drop("item_count", "max_count_in_cluster")

    def _clear_cache(self):
        if hasattr(self, "item_rel_in_cluster"):
            self.item_rel_in_cluster.unpersist()

    @property
    def _dataframes(self):
        return {"item_rel_in_cluster": self.item_rel_in_cluster}

    @staticmethod
    def _transform_features(user_features):
        feature_columns = user_features.drop("user_idx").columns
        vec = VectorAssembler(inputCols=feature_columns, outputCol="features")
        return vec.transform(user_features).select("user_idx", "features")

    def _make_user_clusters(self, users, user_features):

        usr_cnt_in_fv = (
            user_features.select("user_idx")
            .distinct()
            .join(users.distinct(), on="user_idx")
            .count()
        )

        user_cnt = users.distinct().count()

        if usr_cnt_in_fv < user_cnt:
            self.logger.info(
                "% user(s) don't "
                "have a feature vector. "
                "The results will not be calculated for them.",
                user_cnt - usr_cnt_in_fv,
            )

        user_features_vector = self._transform_features(
            user_features.join(users, on="user_idx")
        )
        return (
            self.model.transform(user_features_vector)
            .select("user_idx", "prediction")
            .withColumnRenamed("prediction", "cluster")
        )

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:

        user_clusters = self._make_user_clusters(users, user_features)
        filtered_items = self.item_rel_in_cluster.join(items, on="item_idx")
        pred = user_clusters.join(filtered_items, on="cluster").drop("cluster")
        return pred

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:

        if not user_features:
            raise ValueError("User features are missing for predict")

        user_clusters = self._make_user_clusters(
            pairs.select("user_idx").distinct(), user_features
        )
        pairs_with_clusters = pairs.join(user_clusters, on="user_idx")
        filtered_items = self.item_rel_in_cluster.join(
            pairs.select("item_idx").distinct(), on="item_idx"
        )
        pred = pairs_with_clusters.join(
            filtered_items, on=["cluster", "item_idx"]
        ).select("user_idx", "item_idx", "relevance")
        return pred
