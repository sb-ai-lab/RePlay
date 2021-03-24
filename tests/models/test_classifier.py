# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np
from pyspark.ml.classification import RandomForestClassifier

from replay.models import ClassifierRec
from tests.utils import spark


@pytest.fixture
def log(spark):
    return spark.createDataFrame([("1", "1", 1.0), ("1", "2", 0.0)],).toDF(
        "user_id", "item_id", "relevance"
    )


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame([("1", 1.0, 2.0)]).toDF(
        "user_id", "user_feature_1", "user_feature_2"
    )


@pytest.fixture
def item_features(spark):
    return spark.createDataFrame([("1", 3.0, 4.0), ("2", 5.0, 6.0)]).toDF(
        "item_id", "item_feature_1", "item_feature_2"
    )


@pytest.fixture
def model():
    return ClassifierRec(RandomForestClassifier(seed=47))


def test_fit(log, user_features, item_features, model):
    model.fit(log, user_features, item_features)
    np.allclose(model.model.treeWeights, 20 * [1.0])


def test_predict(log, user_features, item_features, model):
    model.fit(log, user_features, item_features)
    empty_prediction = model.predict(
        log=log,
        k=2,
        users=user_features.select("user_id"),
        items=item_features.select("item_id"),
        user_features=user_features,
        item_features=item_features,
        filter_seen_items=True,
    )
    assert empty_prediction.count() == 0
