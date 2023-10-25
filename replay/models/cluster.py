from typing import Optional

from pandas import DataFrame
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as sf
from replay.data.dataset import Dataset

from replay.models.base_rec import UserRecommender


class ClusterRec(UserRecommender):
    """
    Generate recommendations for cold users using k-means clusters
    """

    can_predict_cold_users = True
    _search_space = {
        "num_clusters": {"type": "int", "args": [2, 20]},
    }
    item_rel_in_cluster: DataFrame

    def __init__(self, num_clusters: int = 10):
        """
        :param num_clusters: number of clusters
        """
        self.num_clusters = num_clusters

    @property
    def _init_args(self):
        return {"num_clusters": self.num_clusters}

    def _save_model(self, path: str):
        self.model.write().overwrite().save(path)

    def _load_model(self, path: str):
        self.model = KMeansModel.load(path)

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        kmeans = KMeans().setK(self.num_clusters).setFeaturesCol("features")
        user_features_vector = self._transform_features(dataset.query_features)
        self.model = kmeans.fit(user_features_vector)
        users_clusters = (
            self.model.transform(user_features_vector)
            .select(self.query_col, "prediction")
            .withColumnRenamed("prediction", "cluster")
        )

        interactions = dataset.interactions.join(users_clusters, on=self.query_col, how="left")
        self.item_rel_in_cluster = interactions.groupBy(["cluster", self.item_col]).agg(
            sf.count(self.item_col).alias("item_count")
        )

        max_count_per_cluster = self.item_rel_in_cluster.groupby(
            "cluster"
        ).agg(sf.max("item_count").alias("max_count_in_cluster"))
        self.item_rel_in_cluster = self.item_rel_in_cluster.join(
            max_count_per_cluster, on="cluster"
        )
        self.item_rel_in_cluster = self.item_rel_in_cluster.withColumn(
            self.rating_col, sf.col("item_count") / sf.col("max_count_in_cluster")
        ).drop("item_count", "max_count_in_cluster")
        self.item_rel_in_cluster.cache().count()

    def _clear_cache(self):
        if hasattr(self, "item_rel_in_cluster"):
            self.item_rel_in_cluster.unpersist()

    @property
    def _dataframes(self):
        return {"item_rel_in_cluster": self.item_rel_in_cluster}

    # @staticmethod
    def _transform_features(self, user_features):
        feature_columns = user_features.drop(self.query_col).columns
        vec = VectorAssembler(inputCols=feature_columns, outputCol="features")
        return vec.transform(user_features).select(self.query_col, "features")

    def _make_user_clusters(self, users, user_features):

        usr_cnt_in_fv = (user_features
                         .select(self.query_col)
                         .distinct()
                         .join(users.distinct(), on=self.query_col).count())

        user_cnt = users.distinct().count()

        if usr_cnt_in_fv < user_cnt:
            self.logger.info("% user(s) don't "
                             "have a feature vector. "
                             "The results will not be calculated for them.",
                             user_cnt - usr_cnt_in_fv)

        user_features_vector = self._transform_features(
            user_features.join(users, on=self.query_col)
        )
        return (
            self.model.transform(user_features_vector)
            .select(self.query_col, "prediction")
            .withColumnRenamed("prediction", "cluster")
        )

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        dataset: Dataset,
        k: int,
        users: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:

        user_clusters = self._make_user_clusters(users, dataset.query_features)
        filtered_items = self.item_rel_in_cluster.join(items, on=self.item_col)
        pred = user_clusters.join(filtered_items, on="cluster").drop("cluster")
        return pred

    def _predict_pairs(
        self,
        pairs: DataFrame,
        dataset: Optional[Dataset] = None,
    ) -> DataFrame:

        if not dataset.query_features:
            raise ValueError("User features are missing for predict")

        user_clusters = self._make_user_clusters(pairs.select(self.query_col).distinct(), dataset.query_features)
        pairs_with_clusters = pairs.join(user_clusters, on=self.query_col)
        filtered_items = (self.item_rel_in_cluster
                          .join(pairs.select(self.item_col).distinct(), on=self.item_col))
        pred = (pairs_with_clusters
                .join(filtered_items, on=["cluster", self.item_col])
                .select(self.query_col,self.item_col,self.rating_col))
        return pred
