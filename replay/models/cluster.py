from typing import Optional

from pandas import DataFrame
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as sf

from replay.models.base_rec import UserRecommender


class ClusterRec(UserRecommender):
    """
    Generate recommendations for cold users using k-means clusters
    """

    can_predict_cold_users = True
    _search_space = {
        "n": {"type": "int", "args": [2, 20]},
    }
    item_rel_in_cluster: DataFrame

    def __init__(self, num_clusters: int = 10):
        """
        :param num_clusters: number of clusters
        """
        self.num_clusters = num_clusters

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:

        kmeans = KMeans().setK(self.num_clusters).setFeaturesCol("features")
        user_features_vector = self._transform_features(user_features)
        self.model = kmeans.fit(user_features_vector)
        users_clusters = (
            self.model.transform(user_features_vector)
            .select("user_idx", "prediction")
            .withColumnRenamed("prediction", "cluster")
        )

        log = log.join(users_clusters, on="user_idx", how="left")
        self.item_rel_in_cluster = log.groupBy(["cluster", "item_idx"]).agg(
            sf.count("item_idx").alias("item_count")
        )

        max_count_per_cluster = self.item_rel_in_cluster.groupby(
            "cluster"
        ).agg(sf.max("item_count").alias("max_count_in_cluster"))
        self.item_rel_in_cluster = self.item_rel_in_cluster.join(
            max_count_per_cluster, on="cluster"
        )
        self.item_rel_in_cluster = self.item_rel_in_cluster.withColumn(
            "relevance", sf.col("item_count") / sf.col("max_count_in_cluster")
        ).drop("item_count", "max_count_in_cluster")
        self.item_rel_in_cluster.cache()

    def _clear_cache(self):
        if hasattr(self, "item_rel_in_cluster"):
            self.item_rel_in_cluster.unpersist()

    @staticmethod
    def _transform_features(user_features):
        feature_columns = user_features.drop("user_idx").columns
        vec = VectorAssembler(inputCols=feature_columns, outputCol="features")
        return vec.transform(user_features).select("user_idx", "features")

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

        user_features_vector = self._transform_features(user_features)
        user_clusters = (
            self.model.transform(user_features_vector)
            .select("user_idx", "prediction")
            .withColumnRenamed("prediction", "cluster")
        )
        filtered_items = self.item_rel_in_cluster.join(items, on="item_idx")
        pred = user_clusters.join(filtered_items, on="cluster").drop("cluster")
        return pred
