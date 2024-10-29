import logging
from datetime import datetime

import pytest

from replay.data import FeatureHint, FeatureInfo, FeatureSchema, FeatureSource, FeatureType, get_schema
from replay.models import LinUCB
from replay.utils import PYSPARK_AVAILABLE
from tests.utils import create_dataset

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf
    from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

    INTERACTIONS_SCHEMA = get_schema("user_idx", "item_idx", "timestamp", "rating")


@pytest.fixture(scope="module")
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, 0, date, 4.0],
            [1, 0, date, 3.0],
            [2, 1, date, 2.0],
            [0, 1, date, 3.0],
            [1, 2, date, 4.0],
            [2, 2, date, 4.0],
            [0, 3, date, 5.0],
            [0, 5, date, 4.0],
            [1, 4, date, 3.0],
        ],
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture(scope="module")
def user_features(spark):
    USER_FEATURES_SCHEMA = StructType(
        [
            StructField("user_idx", IntegerType(), False),
            StructField("user_feature_1", DoubleType(), True),
            StructField("user_feature_2", DoubleType(), True),
        ]
    )
    return spark.createDataFrame(
        [(0, 2.0, 5.0), (1, 0.0, -5.0), (2, 4.0, 3.0)], schema=USER_FEATURES_SCHEMA,
    )


@pytest.fixture(scope="module")
def item_features(spark):
    ITEM_FEATURES_SCHEMA = StructType(
        [
            StructField("item_idx", IntegerType(), False),
            StructField("item_feature_1", DoubleType(), True),
            StructField("item_feature_2", DoubleType(), True),
        ]
    )
    return spark.createDataFrame(
        [(0, 4.0, 5.0), (1, 5.0, 4.0), (2, 2.0, 3.0), (3, 3.0, -1.0), (4, 4.0, 3.0), (5, 3.0, -1.0)],
        schema=ITEM_FEATURES_SCHEMA,
    )

@pytest.fixture(scope="module")
def feature_schema_linucb():
    feature_schema = FeatureSchema([
        FeatureInfo(column="user_idx", feature_type=FeatureType.CATEGORICAL, feature_hint=FeatureHint.QUERY_ID),
        FeatureInfo(column="item_idx", feature_type=FeatureType.CATEGORICAL, feature_hint=FeatureHint.ITEM_ID),
        FeatureInfo(column="rating", feature_type=FeatureType.NUMERICAL, feature_hint=FeatureHint.RATING),
        *[
            FeatureInfo(column=name, feature_type=FeatureType.NUMERICAL, feature_source=FeatureSource.ITEM_FEATURES)
            for name in ["item_feature_1", "item_feature_2"]
        ],
        *[
            FeatureInfo(column=name, feature_type=FeatureType.NUMERICAL, feature_source=FeatureSource.QUERY_FEATURES)
            for name in ["user_feature_1", "user_feature_2"]
        ],
    ])
    return feature_schema


@pytest.fixture(scope="module")
def dataset_linucb(log, user_features, item_features, feature_schema_linucb):
    log_linucb = log.withColumn("rating", sf.when(sf.col("rating") > 3, 1).otherwise(0))

    dataset_lincub = create_dataset(log_linucb, user_features, item_features, feature_schema_linucb)
    return dataset_lincub


@pytest.fixture(scope="module")
def empty_dataset_linucb(log, user_features, item_features, feature_schema_linucb):
    empty_dataset_linucb = create_dataset(log.limit(0), user_features, item_features, feature_schema_linucb)

    return empty_dataset_linucb

@pytest.fixture(params=[LinUCB(eps=-10.0, alpha=1.0, is_hybrid=False)], scope="module")
def fitted_model_disjoint(request, dataset_linucb):
    model = request.param
    model.fit(dataset_linucb)
    return model


@pytest.fixture(params=[LinUCB(eps=-10.0, alpha=1.0, is_hybrid=True)], scope="module")
def fitted_model_hybrid(request, dataset_linucb):
    model = request.param
    model.fit(dataset_linucb)
    return model


@pytest.mark.spark
def test_predict_empty_log_disjoint(fitted_model_disjoint, user_features, empty_dataset_linucb):
    users = user_features.select("user_idx").distinct()
    pred_empty = fitted_model_disjoint.predict(empty_dataset_linucb, queries=users, k=1)
    assert pred_empty.count() == users.count()


@pytest.mark.spark
def test_predict_empty_log_hybrid(fitted_model_hybrid, user_features, empty_dataset_linucb):
    users = user_features.select("user_idx").distinct()
    pred_empty = fitted_model_hybrid.predict(empty_dataset_linucb, queries=users, k=1)
    assert pred_empty.count() == users.count()


@pytest.mark.spark
def test_optimize_disjoint(fitted_model_disjoint, dataset_linucb, caplog):
    with caplog.at_level(logging.WARNING):
        fitted_model_disjoint.optimize(
            dataset_linucb,
            dataset_linucb,
            {"eps": [-20.0, 10.0], "alpha": [0.001, 10.0]},
            k=1,
            budget=1,
        )


@pytest.mark.spark
def test_optimize_hybrid(fitted_model_hybrid, dataset_linucb, caplog):
    with caplog.at_level(logging.WARNING):
        fitted_model_hybrid.optimize(
            dataset_linucb,
            dataset_linucb,
            {"eps": [-20.0, 10.0], "alpha": [0.001, 10.0]},
            k=1,
            budget=1,
        )


@pytest.mark.parametrize("k", [1, 2], ids=["k=1", "k=2"])
@pytest.mark.spark
def test_predict_k_disjoint(fitted_model_disjoint, user_features, dataset_linucb, k):
    users = user_features.select("user_idx").distinct()
    pred = fitted_model_disjoint.predict(dataset_linucb, queries=users, k=k)
    assert pred.count() == users.count() * k


@pytest.mark.parametrize("k", [1, 2], ids=["k=1", "k=2"])
@pytest.mark.spark
def test_predict_k_hybrid(fitted_model_hybrid, user_features, dataset_linucb, k):
    users = user_features.select("user_idx").distinct()
    pred = fitted_model_hybrid.predict(dataset_linucb, queries=users, k=k)
    assert pred.count() == users.count() * k