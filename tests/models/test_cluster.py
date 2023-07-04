# pylint: disable-all

import pandas as pd
import pytest
import pyspark.sql.functions as sf

from replay.model_handler import save, load
from replay.models import ClusterRec
from replay.utils import convert2spark
from tests.utils import (
    spark,
    long_log_with_features,
    short_log_with_features,
    user_features,
    numeric_user_features,
    sparkDataFrameEqual,
)


@pytest.fixture
def model():
    return ClusterRec()


@pytest.fixture
def users_features(spark, user_features):
    return user_features.drop("gender")


def test_works(
    long_log_with_features, model, short_log_with_features, users_features
):
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


def test_cold_user(long_log_with_features, model, users_features):
    train = long_log_with_features.filter("user_idx < 2")
    model.fit(train, user_features=users_features)
    res = model.predict(
        users_features, 2, users=convert2spark(pd.DataFrame({"user_idx": [1]}))
    )
    assert res.count() == 2
    assert res.select("user_idx").distinct().collect()[0][0] == 1
    assert res.filter(sf.col("relevance").isNull()).count() == 0


def test_raises(long_log_with_features, model, users_features):
    with pytest.raises(
        TypeError, match="missing 1 required positional argument"
    ):
        model.fit(long_log_with_features, user_features=users_features)
        model.predict_pairs(
            long_log_with_features.filter(sf.col("user_idx") == 1).select(
                "user_idx", "item_idx"
            )
        )


def test_predict_pairs(long_log_with_features, model, users_features):
    model.fit(long_log_with_features, users_features)

    pairs_pred_k = model.predict_pairs(
        pairs=long_log_with_features.select("user_idx", "item_idx"),
        log=long_log_with_features,
        user_features=users_features,
        k=1,
    )

    pairs_pred = model.predict_pairs(
        pairs=long_log_with_features.select("user_idx", "item_idx"),
        log=long_log_with_features,
        user_features=users_features,
        k=None,
    )

    assert (
        pairs_pred_k.groupBy("user_idx")
        .count()
        .filter(sf.col("count") > 1)
        .count()
        == 0
    )

    assert (
        pairs_pred.groupBy("user_idx")
        .count()
        .filter(sf.col("count") > 1)
        .count()
        > 0
    )


def test_save_load(long_log_with_features, model, numeric_user_features, tmp_path):
    path = (tmp_path / "cluster").resolve()
    model.fit(long_log_with_features, numeric_user_features)
    base_pred = model.predict(numeric_user_features, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(numeric_user_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)
