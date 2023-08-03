# pylint: disable-all

import pandas as pd
import pytest
import pyspark.sql.functions as sf

from replay.models import ClusterRec
from replay.utils.spark_utils import convert2spark
from tests.utils import (
    spark,
    long_log_with_features,
    short_log_with_features,
    user_features,
    sparkDataFrameEqual,
)


@pytest.fixture
def users_features(spark, user_features):
    return user_features.drop("gender")


def test_works(
    long_log_with_features, short_log_with_features, users_features
):
    model = ClusterRec()
    model.fit(long_log_with_features, users_features)
    model.predict(users_features, k=1)
    res = model.optimize(
        long_log_with_features,
        short_log_with_features,
        users_features,
        k=1,
        budget=1,
    )
    assert type(res["num_clusters"]) == int


def test_cold_user(long_log_with_features, users_features):
    model = ClusterRec(2)
    train = long_log_with_features.filter("user_idx < 2")
    model.fit(train, user_features=users_features)
    res = model.predict(
        users_features, 2, users=convert2spark(pd.DataFrame({"user_idx": [1]}))
    )
    assert res.count() == 2
    assert res.select("user_idx").distinct().collect()[0][0] == 1
    assert res.filter(sf.col("relevance").isNull()).count() == 0


def test_predict_pairs(long_log_with_features, users_features):
    model = ClusterRec()
    model.fit(long_log_with_features, user_features=users_features)
    pairs = long_log_with_features.select("user_idx", "item_idx").filter(
        sf.col("user_idx") == 1
    )
    res = model.predict_pairs(
        pairs,
        log=long_log_with_features,
        user_features=users_features,
    )
    sparkDataFrameEqual(res.select("user_idx", "item_idx"), pairs)
    assert res.count() == 4
    assert res.select("user_idx").collect()[0][0] == 1


def test_raises(long_log_with_features, users_features):
    model = ClusterRec()
    with pytest.raises(
        TypeError, match="missing 1 required positional argument"
    ):
        model.fit(long_log_with_features, user_features=users_features)
        model.predict_pairs(
            long_log_with_features.filter(sf.col("user_idx") == 1).select(
                "user_idx", "item_idx"
            )
        )
