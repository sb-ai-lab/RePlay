import pytest

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

from pyspark.sql import functions as sf

from replay.experimental.models import ULinUCB


@pytest.fixture(scope="module")
def log_u_lin_ucb(log2):
    return log2.withColumn("relevance", sf.when(sf.col("relevance") > 3, 1).otherwise(0))


@pytest.fixture(scope="module")
def item_features(spark):
    return spark.createDataFrame([(0, 4.0, 5.0), (1, 5.0, 4.0), (2, 5.0, 1.0), (3, 0.0, 4.0)]).toDF(
        "item_idx", "item_feature_1", "item_feature_2"
    )


@pytest.fixture(scope="module")
def fitted_model(log_u_lin_ucb):
    model = ULinUCB()
    model.fit(log_u_lin_ucb, user_features=None, item_features=item_features)
    return model


@pytest.mark.spark
def test_predict_empty_log(fitted_model, log_u_lin_ucb, seed):
    fitted_model.seed = seed

    users = log_u_lin_ucb.select("user_idx").distinct()
    pred = fitted_model._predict(
        log_u_lin_ucb.limit(0), k=1, users=users, items=list(range(10)), user_features=None, item_features=item_features
    )

    assert pred.count() == users.count()
