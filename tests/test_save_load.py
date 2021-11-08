# pylint: disable-all
from os.path import dirname, join

import pytest
import pandas as pd

from implicit.als import AlternatingLeastSquares
from pyspark.sql import functions as sf

import replay
from replay.model_handler import save, load
from replay.models import *
from tests.utils import sparkDataFrameEqual, long_log_with_features, spark


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame(
        [("u1", 20.0, -3.0, 1), ("u2", 30.0, 4.0, 0), ("u3", 40.0, 0.0, 1)]
    ).toDF("user_id", "age", "mood", "gender")


@pytest.fixture
def df():
    folder = dirname(replay.__file__)
    return pd.read_csv(
        join(folder, "../experiments/data/ml1m_ratings.dat"),
        sep="\t",
        names=["user_id", "item_id", "relevance", "timestamp"],
    ).head(1000)


@pytest.mark.parametrize(
    "recommender",
    [
        ALSWrap,
        ADMMSLIM,
        KNN,
        MultVAE,
        NeuroMF,
        PopRec,
        SLIM,
        UserPopRec,
        LightFMWrap,
    ],
)
def test_equal_preds(long_log_with_features, recommender, tmp_path):
    path = (tmp_path / "test").resolve()
    model = recommender()
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    m = load(path)
    new_pred = m.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_random(long_log_with_features, tmp_path):
    path = (tmp_path / "random").resolve()
    model = RandomRec(seed=1)
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    m = load(path)
    new_pred = m.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_rules(df, tmp_path):
    path = (tmp_path / "rules").resolve()
    model = AssociationRulesItemRec()
    model.fit(df)
    base_pred = model.get_nearest_items(["i1"], 5, metric="lift")
    save(model, path)
    m = load(path)
    new_pred = m.get_nearest_items(["i1"], 5, metric="lift")
    sparkDataFrameEqual(base_pred, new_pred)


def test_word(df, tmp_path):
    path = (tmp_path / "word").resolve()
    model = Word2VecRec()
    model.fit(df)
    base_pred = model.predict(df, 5)
    save(model, path)
    m = load(path)
    new_pred = m.predict(df, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_implicit(long_log_with_features, tmp_path):
    path = (tmp_path / "implicit").resolve()
    model = ImplicitWrap(AlternatingLeastSquares())
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    m = load(path)
    new_pred = m.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_cluster(long_log_with_features, user_features, tmp_path):
    path = (tmp_path / "cluster").resolve()
    model = ClusterRec()
    model.fit(long_log_with_features, user_features)
    base_pred = model.predict(user_features, 5)
    save(model, path)
    m = load(path)
    new_pred = m.predict(user_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_wilson(long_log_with_features, tmp_path):
    path = (tmp_path / "wilson").resolve()
    model = Wilson()
    df = long_log_with_features.withColumn(
        "relevance", (sf.col("relevance") > 3).cast("integer")
    )
    model.fit(df)
    base_pred = model.predict(df, 5)
    save(model, path)
    m = load(path)
    new_pred = m.predict(df, 5)
    sparkDataFrameEqual(base_pred, new_pred)
