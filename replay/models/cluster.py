from typing import Optional

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf

from replay.models.base_rec import UserRecommender, PartialFitMixin
from replay.utils import unionify, unpersist_after


class ClusterRec(UserRecommender, PartialFitMixin):
    """
    Generate recommendations for cold users using k-means clusters
    """

    can_predict_cold_users = True
    _search_space = {
        "num_clusters": {"type": "int", "args": [2, 20]},
    }

    def __init__(
        self, num_clusters: int = 10, hnswlib_params: Optional[dict] = None
    ):
        """
        :param num_clusters: number of clusters
        """
        self.num_clusters = num_clusters
        self._hnswlib_params = hnswlib_params
        self.model: Optional[KMeansModel] = None
        self.users_clusters: Optional[DataFrame] = None
        self.item_rel_in_cluster: Optional[DataFrame] = None
        self.item_count_in_cluster: Optional[DataFrame] = None

    @property
    def _init_args(self):
        return {"num_clusters": self.num_clusters}

    def _save_model(self, path: str):
        self.model.write().overwrite().save(path)

    def _load_model(self, path: str):
        self.model = KMeansModel.load(path)

    def _get_nearest_items(self, items: DataFrame, metric: Optional[str] = None,
                           candidates: Optional[DataFrame] = None) -> Optional[DataFrame]:
        raise NotImplementedError()

    def _fit_partial(self, log: DataFrame, user_features: Optional[DataFrame] = None,) -> None:
        with unpersist_after(self._dataframes):
            user_features_vector = self._transform_features(user_features) if user_features is not None else None

            if self.model is None:
                assert user_features_vector is not None
                self.model = KMeans().setK(self.num_clusters).setFeaturesCol("features").fit(user_features_vector)

            if user_features_vector is not None:
                users_clusters = (
                    self.model
                        .transform(user_features_vector)
                        .select("user_idx", sf.col("prediction").alias('cluster'))
                )

                # update if we make fit_partial instead of just fit
                self.users_clusters = unionify(users_clusters, self.users_clusters).drop_duplicates(["user_idx"])

            item_count_in_cluster = (
                log.join(self.users_clusters, on="user_idx", how="left")
                .groupBy(["cluster", "item_idx"])
                .agg(sf.count("item_idx").alias("item_count"))
            )

            # update if we make fit_partial instead of just fit
            self.item_count_in_cluster = (
                unionify(item_count_in_cluster, self.item_count_in_cluster)
                .groupBy(["cluster", "item_idx"])
                .agg(sf.sum("item_count").alias("item_count"))
            )

            self.item_rel_in_cluster = self.item_count_in_cluster.withColumn(
                "relevance", sf.col("item_count") / sf.max("item_count").over(Window.partitionBy("cluster"))
            ).drop("item_count", "max_count_in_cluster").cache()

            # materialize datasets
            self.item_rel_in_cluster.count()

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self._fit_partial(log, user_features)

    def fit_partial(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        previous_log: Optional[DataFrame] = None
    ) -> None:
        self._fit_partial(log, user_features)

    def _clear_cache(self):
        for df in self._dataframes.values():
            if df is not None:
                df.unpersist()

    @property
    def _dataframes(self):
        return {
            "users_clusters": self.users_clusters,
            "item_rel_in_cluster": self.item_rel_in_cluster,
            "item_count_in_cluster": self.item_count_in_cluster
        }

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
