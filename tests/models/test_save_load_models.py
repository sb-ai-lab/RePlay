# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import, wildcard-import, unused-wildcard-import
from os.path import dirname, join

import pandas as pd
import pytest

import replay
from replay.data import FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.models import *
from replay.models.extensions.ann.entities.hnswlib_param import HnswlibParam
from replay.models.extensions.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.models.extensions.ann.index_builders.driver_hnswlib_index_builder import DriverHnswlibIndexBuilder
from replay.models.extensions.ann.index_builders.driver_nmslib_index_builder import DriverNmslibIndexBuilder
from replay.models.extensions.ann.index_builders.executor_nmslib_index_builder import ExecutorNmslibIndexBuilder
from replay.models.extensions.ann.index_stores.hdfs_index_store import HdfsIndexStore
from replay.models.extensions.ann.index_stores.shared_disk_index_store import SharedDiskIndexStore
from replay.preprocessing.label_encoder import LabelEncoder, LabelEncodingRule
from replay.utils import PYSPARK_AVAILABLE
from tests.models.test_cat_pop_rec import cat_log, cat_tree, requested_cats
from tests.utils import create_dataset, long_log_with_features, spark, sparkDataFrameEqual

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf

    from replay.models.extensions.ann.index_stores.spark_files_index_store import SparkFilesIndexStore
    from replay.utils.model_handler import load, save
    from replay.utils.spark_utils import convert2spark


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
        names=["user_idx", "item_idx", "relevance", "timestamp"],
    ).head(1000)
    res = convert2spark(res)
    encoder = LabelEncoder(
        [
            LabelEncodingRule("user_idx"),
            LabelEncodingRule("item_idx"),
        ]
    )
    res = encoder.fit_transform(res)
    return res


@pytest.mark.spark
@pytest.mark.parametrize(
    "recommender",
    [
        ALSWrap,
        ItemKNN,
        PopRec,
        SLIM,
        QueryPopRec,
    ],
)
def test_equal_preds(long_log_with_features, recommender, tmp_path):
    path = (tmp_path / "test").resolve()
    dataset = create_dataset(long_log_with_features)
    model = recommender()
    model.fit(dataset)
    base_pred = model.predict(dataset, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(dataset, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.spark
def test_random(long_log_with_features, tmp_path):
    path = (tmp_path / "random").resolve()
    model = RandomRec(seed=1)
    dataset = create_dataset(long_log_with_features)
    model.fit(dataset)
    base_pred = model.predict(dataset, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(dataset, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.spark
def test_rules(df, tmp_path):
    path = (tmp_path / "rules").resolve()
    dataset = create_dataset(df)
    model = AssociationRulesItemRec(session_column="user_idx")
    model.fit(dataset)
    base_pred = model.get_nearest_items([1], 5, metric="lift")
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.get_nearest_items([1], 5, metric="lift")
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.spark
def test_word(df, tmp_path):
    path = (tmp_path / "word").resolve()
    dataset = create_dataset(df)
    model = Word2VecRec()
    model.fit(dataset)
    base_pred = model.predict(dataset, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(dataset, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.spark
def test_cluster(long_log_with_features, user_features, tmp_path):
    path = (tmp_path / "cluster").resolve()
    dataset = create_dataset(long_log_with_features, user_features)
    model = ClusterRec()
    model.fit(dataset)
    base_pred = model.predict(dataset, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(dataset, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.spark
def test_cat_poprec(cat_tree, cat_log, requested_cats, tmp_path):
    path = (tmp_path / "cat_poprec").resolve()
    feature_schema = FeatureSchema(
        [
            FeatureInfo(
                column="user_idx",
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.QUERY_ID,
            ),
            FeatureInfo(
                column="item_idx",
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.ITEM_ID,
            ),
            FeatureInfo(
                column="category",
                feature_type=FeatureType.CATEGORICAL,
            ),
            FeatureInfo(
                column="relevance",
                feature_type=FeatureType.NUMERICAL,
                feature_hint=FeatureHint.RATING,
            ),
        ]
    )
    dataset = create_dataset(cat_log, feature_schema=feature_schema)
    model = CatPopRec(cat_tree=cat_tree)
    model.fit(dataset)
    base_pred = model.predict(requested_cats, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(requested_cats, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.spark
@pytest.mark.parametrize("model", [Wilson(), UCB()], ids=["wilson", "ucb"])
def test_wilson_ucb(model, log_unary, tmp_path):
    path = (tmp_path / "model").resolve()
    dataset = create_dataset(log_unary)
    model.fit(dataset)
    base_pred = model.predict(dataset, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(dataset, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.spark
def test_study(df, tmp_path):
    path = (tmp_path / "study").resolve()
    dataset = create_dataset(df)
    model = PopRec()
    model.study = 80083
    model.fit(dataset)
    save(model, path)
    loaded_model = load(path)
    assert loaded_model.study == model.study


@pytest.mark.spark
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
    dataset = create_dataset(long_log_with_features)
    model.fit(dataset)
    base_pred = model.predict(dataset, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(dataset, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.spark
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
    dataset = create_dataset(long_log_with_features)
    model.fit(dataset)
    base_pred = model.predict(dataset, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(dataset, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.spark
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
    dataset = create_dataset(long_log_with_features)
    model.fit(dataset)
    base_pred = model.predict(dataset, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(dataset, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.core
def test_hdfs_index_store_exception():
    local_warehouse_dir = 'file:///tmp'
    with pytest.raises(ValueError, match=f"Can't recognize path {local_warehouse_dir + '/index_dir'} as HDFS path!"):
        HdfsIndexStore(warehouse_dir=local_warehouse_dir, index_dir="index_dir")
