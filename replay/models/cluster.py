# pylint: disable=invalid-name, too-many-arguments, attribute-defined-outside-init
from typing import Optional

from pandas import DataFrame
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as sf

from replay.models.base_rec import UserRecommender
from replay.session_handler import State


class ClusterRec(UserRecommender):
    """
    Generate recommendations for cold users using k-means clusters
    """

    can_predict_cold_users = True
    _search_space = {
        "n": {"type": "int", "args": [2, 20]},
    }

    def __init__(self, n: int = 10):
        """
        :param n: number of clusters
        """
        State()  # initialize session if it doesnt exist
        self.n = n

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:

        kmeans = KMeans().setK(self.n).setFeaturesCol("features")
        df = self._transform_features(user_features)
        self.model = kmeans.fit(df)
        df = (
            self.model.transform(df)
            .select("user_idx", "prediction")
            .withColumnRenamed("prediction", "cluster")
        )

        log = log.join(df, on="user_idx", how="left")
        log = log.groupBy(["cluster", "item_idx"]).agg(
            sf.count("item_idx").alias("count")
        )
        max_count_per_cluster = log.groupby("cluster").agg(
            sf.max("count").alias("max")
        )
        log = log.join(max_count_per_cluster, on="cluster", how="left")
        log = log.withColumn(
            "relevance", sf.col("count") / sf.col("max")
        ).drop("count", "max")
        self.recs = log

    @staticmethod
    def _transform_features(df):
        feature_columns = df.drop("user_idx").columns
        vec = VectorAssembler(inputCols=feature_columns, outputCol="features")
        return vec.transform(df).select("user_idx", "features")

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

        df = self._transform_features(user_features)
        df = (
            self.model.transform(df)
            .select("user_idx", "prediction")
            .withColumnRenamed("prediction", "cluster")
        )
        pred = df.join(self.recs, on="cluster").drop("cluster")
        return pred
