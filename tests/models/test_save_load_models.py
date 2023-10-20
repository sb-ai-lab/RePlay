# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import, wildcard-import, unused-wildcard-import
import os
from functools import partial
from os.path import dirname, join

import pytest
import pandas as pd

from implicit.als import AlternatingLeastSquares
from pyspark.sql import functions as sf

import replay
from replay.models.extensions.ann.entities.hnswlib_param import HnswlibParam
from replay.models.extensions.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.models.extensions.ann.index_builders.driver_hnswlib_index_builder import DriverHnswlibIndexBuilder
from replay.models.extensions.ann.index_builders.driver_nmslib_index_builder import (
    DriverNmslibIndexBuilder,
)
from replay.models.extensions.ann.index_builders.executor_nmslib_index_builder import (
    ExecutorNmslibIndexBuilder,
)
from replay.models.extensions.ann.index_builders.executor_hnswlib_index_builder import (
    ExecutorHnswlibIndexBuilder,
)
from replay.models.extensions.ann.index_stores.hdfs_index_store import HdfsIndexStore
from replay.models.extensions.ann.index_stores.shared_disk_index_store import (
    SharedDiskIndexStore,
)
from replay.models.extensions.ann.index_stores.spark_files_index_store import (
    SparkFilesIndexStore,
)
from replay.preprocessing.data_preparator import Indexer
from replay.utils.model_handler import save, load
from replay.models import *
from replay.utils.spark_utils import convert2spark
from tests.utils import long_log_with_features, sparkDataFrameEqual, spark
from tests.models.test_cat_pop_rec import cat_tree, cat_log, requested_cats


@pytest.fixture
def log_unary(long_log_with_features):
    return long_log_with_features.withColumn(
        "relevance", sf.when(sf.col("relevance") > 3, 1).otherwise(0)
    )


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame(
        [(1, 20.0, -3.0, 1), (2, 30.0, 4.0, 0), (3, 40.0, 0.0, 1)]
    ).toDF("user_idx", "age", "mood", "gender")


@pytest.fixture
def df():
    folder = dirname(replay.__file__)
    res = pd.read_csv(
        join(folder, "../experiments/data/ml1m_ratings.dat"),
        sep="\t",
        names=["user_id", "item_id", "relevance", "timestamp"],
    ).head(1000)
    res = convert2spark(res)
    indexer = Indexer()
    indexer.fit(res, res)
    res = indexer.transform(res)
    return res


@pytest.mark.parametrize(
    "recommender",
    [
        ALSWrap,
        ItemKNN,
        PopRec,
        SLIM,
        UserPopRec,
    ],
)
def test_equal_preds(long_log_with_features, recommender, tmp_path):
    path = (tmp_path / "test").resolve()
    model = recommender()
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_random(long_log_with_features, tmp_path):
    path = (tmp_path / "random").resolve()
    model = RandomRec(seed=1)
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_rules(df, tmp_path):
    path = (tmp_path / "rules").resolve()
    model = AssociationRulesItemRec()
    model.fit(df)
    base_pred = model.get_nearest_items([1], 5, metric="lift")
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.get_nearest_items([1], 5, metric="lift")
    sparkDataFrameEqual(base_pred, new_pred)


def test_word(df, tmp_path):
    path = (tmp_path / "word").resolve()
    model = Word2VecRec()
    model.fit(df)
    base_pred = model.predict(df, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(df, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_cluster(long_log_with_features, user_features, tmp_path):
    path = (tmp_path / "cluster").resolve()
    model = ClusterRec()
    model.fit(long_log_with_features, user_features)
    base_pred = model.predict(user_features, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(user_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_cat_poprec(cat_tree, cat_log, requested_cats, tmp_path):
    path = (tmp_path / "cat_poprec").resolve()
    model = CatPopRec(cat_tree=cat_tree)
    model.fit(cat_log)
    base_pred = model.predict(requested_cats, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(requested_cats, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.parametrize("model", [Wilson(), UCB()], ids=["wilson", "ucb"])
def test_wilson_ucb(model, log_unary, tmp_path):
    path = (tmp_path / "model").resolve()
    model.fit(log_unary)
    base_pred = model.predict(log_unary, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(log_unary, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_study(df, tmp_path):
    path = (tmp_path / "study").resolve()
    model = PopRec()
    model.study = 80083
    model.fit(df)
    save(model, path)
    loaded_model = load(path)
    assert loaded_model.study == model.study


def test_ann_word2vec_saving_loading(long_log_with_features, tmp_path):
    model = Word2VecRec(
        rank=1, window_size=1, use_idf=True, seed=42, min_count=0,
        index_builder=DriverHnswlibIndexBuilder(
            index_params=HnswlibParam(
                space="l2",
                m=100,
                ef_c=2000,
                post=0,
                ef_s=2000,
            ),
            index_store=SharedDiskIndexStore(
                warehouse_dir=str(tmp_path),
                index_dir="hnswlib_index"
            )
        )
    )

    path = (tmp_path / "test").resolve()
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_ann_slim_saving_loading(long_log_with_features, tmp_path):
    nmslib_hnsw_params = NmslibHnswParam(
        space="negdotprod_sparse",
        m=10,
        ef_s=200,
        ef_c=200,
        post=0,
    )
    model = SLIM(
        0.0,
        0.01,
        seed=42,
        index_builder=DriverNmslibIndexBuilder(
            index_params=nmslib_hnsw_params,
            index_store=SparkFilesIndexStore(),
        ),
    )

    path = (tmp_path / "test").resolve()
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_ann_knn_saving_loading(long_log_with_features, tmp_path):
    nmslib_hnsw_params = NmslibHnswParam(
        space="negdotprod_sparse",
        m=10,
        ef_s=200,
        ef_c=200,
        post=0,
    )
    model = ItemKNN(
        1,
        weighting=None,
        index_builder=ExecutorNmslibIndexBuilder(
            index_params=nmslib_hnsw_params,
            index_store=SharedDiskIndexStore(
                warehouse_dir=str(tmp_path), index_dir="nmslib_hnsw_index"
            ),
        ),
    )

    path = (tmp_path / "test").resolve()
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


def test_hdfs_index_store_exception():
    local_warehouse_dir = 'file:///tmp'
    with pytest.raises(ValueError, match=f"Can't recognize path {local_warehouse_dir + '/index_dir'} as HDFS path!"):
        HdfsIndexStore(warehouse_dir=local_warehouse_dir, index_dir="index_dir")
