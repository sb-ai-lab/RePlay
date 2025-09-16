import pandas as pd
import pytest

from replay.models import ClusterRec
from tests.utils import (
    create_dataset,
    sparkDataFrameEqual,
)

pyspark = pytest.importorskip("pyspark")
import pyspark.sql.functions as sf

from replay.utils.spark_utils import convert2spark


@pytest.fixture(scope="module")
def users_features(user_features):
    return user_features.drop("gender")


@pytest.mark.conditional
def test_works(long_log_with_features, short_log_with_features, users_features):
    model = ClusterRec()
    train_dataset = create_dataset(long_log_with_features, user_features=users_features)
    test_dataset = create_dataset(short_log_with_features, user_features=users_features)
    model.fit(train_dataset)
    model.predict(train_dataset, k=1)
    res = model.optimize(
        train_dataset,
        test_dataset,
        k=1,
        budget=1,
    )
    assert isinstance(res["num_clusters"], int)


@pytest.mark.spark
def test_cold_user(long_log_with_features, users_features):
    model = ClusterRec(2)
    train = long_log_with_features.filter("user_idx < 2")
    train_dataset = create_dataset(train, users_features)
    model.fit(train_dataset)
    res = model.predict(train_dataset, 2, queries=convert2spark(pd.DataFrame({"user_idx": [1]})))
    assert res.count() == 2
    assert res.select("user_idx").distinct().collect()[0][0] == 1
    assert res.filter(sf.col("relevance").isNull()).count() == 0


@pytest.mark.spark
def test_predict_pairs(long_log_with_features, users_features):
    model = ClusterRec()
    train_dataset = create_dataset(long_log_with_features, users_features)
    model.fit(train_dataset)
    pairs = long_log_with_features.select("user_idx", "item_idx").filter(sf.col("user_idx") == 1)
    res = model.predict_pairs(
        pairs,
        dataset=train_dataset,
    )
    sparkDataFrameEqual(res.select("user_idx", "item_idx"), pairs)
    assert res.count() == 4
    assert res.select("user_idx").collect()[0][0] == 1


@pytest.mark.spark
def test_raises(long_log_with_features, users_features):
    model = ClusterRec()
    with pytest.raises(TypeError, match=r"missing 1 required positional argument"):
        train_dataset = create_dataset(long_log_with_features, users_features)
        model.fit(train_dataset)
        model.predict_pairs(long_log_with_features.filter(sf.col("user_idx") == 1).select("user_idx", "item_idx"))


@pytest.mark.spark
def test_predict_empty_log(long_log_with_features, users_features):
    model = ClusterRec()
    train_dataset = create_dataset(long_log_with_features, user_features=users_features)
    test_dataset = create_dataset(long_log_with_features.limit(0), user_features=users_features)
    model.fit(train_dataset)
    model.predict(test_dataset, k=1)


@pytest.mark.spark
def test_predict_empty_dataset(long_log_with_features, users_features):
    with pytest.raises(ValueError, match=r"Query features are missing for predict"):
        model = ClusterRec()
        train_dataset = create_dataset(long_log_with_features, user_features=users_features)
        model.fit(train_dataset)
        model.predict(None, k=1)

    with pytest.raises(ValueError, match=r"Query features are missing for predict"):
        model = ClusterRec()
        train_dataset = create_dataset(long_log_with_features, user_features=users_features)
        pred_dataset = create_dataset(long_log_with_features, user_features=None)
        model.fit(train_dataset)
        model.predict(pred_dataset, k=1)


@pytest.mark.spark
def test_raise_without_features(long_log_with_features, users_features):
    with pytest.raises(ValueError, match=r"Query features are missing for predict"):
        model = ClusterRec()
        train_dataset = create_dataset(long_log_with_features, user_features=users_features)
        test_dataset = create_dataset(long_log_with_features)
        pairs = long_log_with_features.select("user_idx", "item_idx").filter(sf.col("user_idx") == 1)
        model.fit(train_dataset)
        model.predict_pairs(
            pairs,
            dataset=test_dataset,
        )
