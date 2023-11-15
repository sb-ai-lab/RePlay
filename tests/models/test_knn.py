# pylint: disable-all
from datetime import datetime

import numpy as np
import pytest

from replay.data import Dataset, get_schema
from replay.models import ItemKNN
from replay.models.extensions.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.models.extensions.ann.index_builders.driver_nmslib_index_builder import DriverNmslibIndexBuilder
from replay.utils import PYSPARK_AVAILABLE
from tests.utils import create_dataset, spark

if PYSPARK_AVAILABLE:
    from replay.models.extensions.ann.index_stores.spark_files_index_store import SparkFilesIndexStore
    from replay.utils.spark_utils import convert2spark
    INTERACTIONS_SCHEMA = get_schema("user_idx", "item_idx", "timestamp", "relevance")


@pytest.fixture
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, 0, date, 1.0],
            [1, 1, date, 1.0],
            [2, 0, date, 1.0],
            [2, 1, date, 1.0],
        ],
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture
def log_2items_per_user(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, 0, date, 1.0],
            [0, 1, date, 1.0],
            [1, 0, date, 1.0],
            [1, 1, date, 1.0],
            [2, 2, date, 1.0],
            [2, 3, date, 1.0],
        ],
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture
def weighting_log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, 0, date, 1.0],
            [0, 1, date, 1.0],
            [1, 1, date, 1.0],
            [2, 0, date, 1.0],
            [2, 1, date, 1.0],
        ],
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture
def model():
    model = ItemKNN(1, weighting=None)
    return model


@pytest.fixture
def model_with_ann():
    nmslib_hnsw_params = NmslibHnswParam(
        space="negdotprod_sparse",
        m=10,
        ef_s=200,
        ef_c=200,
        post=0,
    )
    return ItemKNN(
        1,
        weighting=None,
        index_builder=DriverNmslibIndexBuilder(
            index_params=nmslib_hnsw_params, index_store=SparkFilesIndexStore()
        ),
    )


@pytest.fixture
def tf_idf_model():
    model = ItemKNN(1, weighting="tf_idf")
    return model


@pytest.fixture
def bm25_model():
    model = ItemKNN(1, weighting="bm25")
    return model


@pytest.mark.core
def test_invalid_weighting():
    with pytest.raises(ValueError):
        ItemKNN(1, weighting="invalid_weighting")


@pytest.mark.spark
def test_works(log, model):
    dataset = create_dataset(log)
    model.fit(dataset)
    recs = model.predict(dataset, k=1, queries=[0, 1]).toPandas()
    assert recs.loc[recs["user_idx"] == 0, "item_idx"].iloc[0] == 1
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] == 0


@pytest.mark.spark
def test_tf_idf(weighting_log, tf_idf_model):
    train_dataset = create_dataset(weighting_log)
    tf_idf_model.fit(train_dataset)
    idf = tf_idf_model._get_idf(train_dataset.interactions).toPandas()
    assert np.allclose(idf[idf["user_idx"] == 1]["idf"], np.log1p(2 / 1))
    assert np.allclose(idf[idf["user_idx"] == 0]["idf"], np.log1p(2 / 2))
    assert np.allclose(idf[idf["user_idx"] == 2]["idf"], np.log1p(2 / 2))

    tf_idf_model.fit(train_dataset)
    recs = tf_idf_model.predict(train_dataset, k=1, queries=[0, 1]).toPandas()
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] == 0


@pytest.mark.spark
def test_bm25(weighting_log, bm25_model):
    k1 = bm25_model.bm25_k1
    b = bm25_model.bm25_b
    avgdl = (2 + 3) / 2

    train_dataset = create_dataset(weighting_log)
    bm25_model.fit(train_dataset)

    log = bm25_model._get_tf_bm25(train_dataset.interactions).toPandas()
    assert np.allclose(
        log[log["item_idx"] == 1]["relevance"],
        1 * (k1 + 1) / (1 + k1 * (1 - b + b * 3 / avgdl)),
    )
    assert np.allclose(
        log[log["item_idx"] == 0]["relevance"],
        1 * (k1 + 1) / (1 + k1 * (1 - b + b * 2 / avgdl)),
    )

    idf = bm25_model._get_idf(train_dataset.interactions).toPandas()
    assert np.allclose(
        idf[idf["user_idx"] == 1]["idf"],
        np.log1p((2 - 1 + 0.5) / (1 + 0.5)),
    )
    assert np.allclose(
        idf[idf["user_idx"] == 0]["idf"],
        np.log1p((2 - 2 + 0.5) / (2 + 0.5)),
    )
    assert np.allclose(
        idf[idf["user_idx"] == 2]["idf"],
        np.log1p((2 - 2 + 0.5) / (2 + 0.5)),
    )
    upd_dataset = Dataset(
        train_dataset.feature_schema,
        convert2spark(log),
    )
    bm25_model.fit(upd_dataset)
    recs = bm25_model.predict(upd_dataset, k=1, queries=[0, 1]).toPandas()
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] == 0


@pytest.mark.spark
def test_weighting_raises(log, tf_idf_model):
    with pytest.raises(ValueError, match="weighting must be one of .*"):
        tf_idf_model.weighting = " "
        dataset = create_dataset(log)
        tf_idf_model.fit(dataset)
        log = tf_idf_model._reweight_interactions(dataset.interactions)


@pytest.mark.spark
def test_knn_predict_filter_seen_items(log, model, model_with_ann):
    dataset = create_dataset(log)
    model.fit(dataset)
    recs1 = model.predict(dataset, k=1, filter_seen_items=True)

    model_with_ann.fit(dataset)
    recs2 = model_with_ann.predict(dataset, k=1, filter_seen_items=True)

    recs1 = recs1.toPandas().sort_values(
        ["user_idx", "item_idx"], ascending=False
    )
    recs2 = recs2.toPandas().sort_values(
        ["user_idx", "item_idx"], ascending=False
    )
    assert recs1.user_idx.equals(recs2.user_idx)
    assert recs1.item_idx.equals(recs2.item_idx)


@pytest.mark.spark
def test_knn_predict(log_2items_per_user, model, model_with_ann):
    dataset = create_dataset(log_2items_per_user)
    model.fit(dataset)
    recs1 = model.predict(dataset, k=2, filter_seen_items=False)

    model_with_ann.fit(dataset)
    recs2 = model_with_ann.predict(
        dataset, k=2, filter_seen_items=False
    )

    recs1 = recs1.toPandas().sort_values(
        ["user_idx", "item_idx"], ascending=False
    )
    recs2 = recs2.toPandas().sort_values(
        ["user_idx", "item_idx"], ascending=False
    )
    assert all(recs1.user_idx.values == recs2.user_idx.values)
    assert all(recs1.item_idx.values == recs2.item_idx.values)
