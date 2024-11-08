import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark.sql import functions as sf
from sklearn.cluster import KMeans

from replay.experimental.models.hierarchical_recommender import HierarchicalRecommender


@pytest.fixture(scope="module")
def log_u_lin_ucb(log2):
    return log2.withColumn("relevance", sf.when(sf.col("relevance") > 3, 1).otherwise(0))


@pytest.fixture(scope="module")
def user_features(spark):
    return spark.createDataFrame([(0, 2.0, 5.0), (1, 0.0, -5.0), (2, 4.0, 3.0)]).toDF(
        "user_idx", "user_feature_1", "user_feature_2"
    )


@pytest.fixture(scope="module")
def item_features(spark):
    return spark.createDataFrame([(0, 4.0, 5.0), (1, 5.0, 4.0), (2, 5.0, 1.0), (3, 0.0, 4.0)]).toDF(
        "item_idx", "item_feature_1", "item_feature_2"
    )


@pytest.fixture(scope="module")
def fitted_model(log_u_lin_ucb, user_features, item_features, random_state=42):
    model = HierarchicalRecommender(depth=2, cluster_model=KMeans(n_clusters=2, random_state=random_state))
    model.fit(log_u_lin_ucb, user_features=user_features, item_features=item_features)
    return model


def test_predict(fitted_model, log_u_lin_ucb, user_features, item_features):
    users = log_u_lin_ucb.select("user_idx").distinct()
    pred = fitted_model.predict(
        log_u_lin_ucb,
        k=1,
        users=users,
        items=list(range(10)),
        user_features=user_features,
        item_features=item_features,
    )

    assert pred.count() <= users.count()
