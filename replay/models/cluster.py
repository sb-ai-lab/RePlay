from os.path import join
from typing import Optional

from pandas import DataFrame
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as sf
from replay.data.dataset import Dataset

from replay.models.base_rec import QueryRecommender


class ClusterRec(QueryRecommender):
    """
    Generate recommendations for cold queries using k-means clusters
    """

    can_predict_cold_queries = True
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

    def _save_model(self, path: str, additional_params: Optional[dict] = None):
        super()._save_model(path, additional_params)
        self.model.write().overwrite().save(join(path, "model"))

    def _load_model(self, path: str):
        super()._load_model(path)
        self.model = KMeansModel.load(join(path, "model"))

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        kmeans = KMeans().setK(self.num_clusters).setFeaturesCol("features")
        query_features_vector = self._transform_features(dataset.query_features)
        self.model = kmeans.fit(query_features_vector)
        queries_clusters = (
            self.model.transform(query_features_vector)
            .select(self.query_column, "prediction")
            .withColumnRenamed("prediction", "cluster")
        )

        interactions = dataset.interactions.join(queries_clusters, on=self.query_column, how="left")
        self.item_rel_in_cluster = interactions.groupBy(["cluster", self.item_column]).agg(
            sf.count(self.item_column).alias("item_count")
        )

        max_count_per_cluster = self.item_rel_in_cluster.groupby(
            "cluster"
        ).agg(sf.max("item_count").alias("max_count_in_cluster"))
        self.item_rel_in_cluster = self.item_rel_in_cluster.join(
            max_count_per_cluster, on="cluster"
        )
        self.item_rel_in_cluster = self.item_rel_in_cluster.withColumn(
            self.rating_column, sf.col("item_count") / sf.col("max_count_in_cluster")
        ).drop("item_count", "max_count_in_cluster")
        self.item_rel_in_cluster.cache().count()

    def _clear_cache(self):
        if hasattr(self, "item_rel_in_cluster"):
            self.item_rel_in_cluster.unpersist()

    @property
    def _dataframes(self):
        return {"item_rel_in_cluster": self.item_rel_in_cluster}

    def _transform_features(self, query_features):
        feature_columns = query_features.drop(self.query_column).columns
        vec = VectorAssembler(inputCols=feature_columns, outputCol="features")
        return vec.transform(query_features).select(self.query_column, "features")

    def _make_query_clusters(self, queries, query_features):

        query_cnt_in_fv = (
            query_features
            .select(self.query_column)
            .distinct()
            .join(queries.distinct(), on=self.query_column)
            .count()
        )

        query_cnt = queries.distinct().count()

        if query_cnt_in_fv < query_cnt:
            self.logger.info("%s query(s) don't "
                             "have a feature vector. "
                             "The results will not be calculated for them.",
                             query_cnt - query_cnt_in_fv)

        query_features_vector = self._transform_features(
            query_features.join(queries, on=self.query_column)
        )
        return (
            self.model.transform(query_features_vector)
            .select(self.query_column, "prediction")
            .withColumnRenamed("prediction", "cluster")
        )

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        dataset: Dataset,
        k: int,
        queries: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True,
    ) -> DataFrame:

        query_clusters = self._make_query_clusters(queries, dataset.query_features)
        filtered_items = self.item_rel_in_cluster.join(items, on=self.item_column)
        pred = query_clusters.join(filtered_items, on="cluster").drop("cluster")
        return pred

    def _predict_pairs(
        self,
        pairs: DataFrame,
        dataset: Optional[Dataset] = None,
    ) -> DataFrame:

        if not dataset.query_features:
            raise ValueError("Query features are missing for predict")

        query_clusters = self._make_query_clusters(pairs.select(self.query_column).distinct(), dataset.query_features)
        pairs_with_clusters = pairs.join(query_clusters, on=self.query_column)
        filtered_items = (self.item_rel_in_cluster
                          .join(pairs.select(self.item_column).distinct(), on=self.item_column))
        pred = (pairs_with_clusters
                .join(filtered_items, on=["cluster", self.item_column])
                .select(self.query_column,self.item_column,self.rating_column))
        return pred
