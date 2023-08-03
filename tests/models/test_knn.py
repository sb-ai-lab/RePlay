# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np

from replay.models.extensions.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.models.extensions.ann.index_builders.driver_nmslib_index_builder import (
    DriverNmslibIndexBuilder,
)
from replay.models.extensions.ann.index_stores.spark_files_index_store import (
    SparkFilesIndexStore,
)
from replay.data import LOG_SCHEMA
from replay.models import ItemKNN
from tests.utils import spark


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
        schema=LOG_SCHEMA,
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
        schema=LOG_SCHEMA,
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
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def model():
    model = ItemKNN(1, weighting=None)
    return model


@pytest.fixture
def model_with_ann(tmp_path):
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


def test_works(log, model):
    model.fit(log)
    recs = model.predict(log, k=1, users=[0, 1]).toPandas()
    assert recs.loc[recs["user_idx"] == 0, "item_idx"].iloc[0] == 1
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] == 0


def test_tf_idf(weighting_log, tf_idf_model):
    idf = tf_idf_model._get_idf(weighting_log).toPandas()
    assert np.allclose(idf[idf["user_idx"] == 1]["idf"], np.log1p(2 / 1))
    assert np.allclose(idf[idf["user_idx"] == 0]["idf"], np.log1p(2 / 2))
    assert np.allclose(idf[idf["user_idx"] == 2]["idf"], np.log1p(2 / 2))

    tf_idf_model.fit(weighting_log)
    recs = tf_idf_model.predict(weighting_log, k=1, users=[0, 1]).toPandas()
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] == 0


def test_bm25(weighting_log, bm25_model):
    k1 = bm25_model.bm25_k1
    b = bm25_model.bm25_b
    avgdl = (2 + 3) / 2

    log = bm25_model._get_tf_bm25(weighting_log).toPandas()
    assert np.allclose(
        log[log["item_idx"] == 1]["relevance"],
        1 * (k1 + 1) / (1 + k1 * (1 - b + b * 3 / avgdl)),
    )
    assert np.allclose(
        log[log["item_idx"] == 0]["relevance"],
        1 * (k1 + 1) / (1 + k1 * (1 - b + b * 2 / avgdl)),
    )

    idf = bm25_model._get_idf(weighting_log).toPandas()
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

    bm25_model.fit(weighting_log)
    recs = bm25_model.predict(weighting_log, k=1, users=[0, 1]).toPandas()
    assert recs.loc[recs["user_idx"] == 1, "item_idx"].iloc[0] == 0


def test_weighting_raises(log, tf_idf_model):
    with pytest.raises(ValueError, match="weighting must be one of .*"):
        tf_idf_model.weighting = " "
        log = tf_idf_model._reweight_log(log)


def test_knn_predict_filter_seen_items(log, model, model_with_ann):
    model.fit(log)
    recs1 = model.predict(log, k=1, filter_seen_items=True)

    model_with_ann.fit(log)
    recs2 = model_with_ann.predict(log, k=1, filter_seen_items=True)

    recs1 = recs1.toPandas().sort_values(
        ["user_idx", "item_idx"], ascending=False
    )
    recs2 = recs2.toPandas().sort_values(
        ["user_idx", "item_idx"], ascending=False
    )
    assert recs1.user_idx.equals(recs2.user_idx)
    assert recs1.item_idx.equals(recs2.item_idx)


def test_knn_predict(log_2items_per_user, model, model_with_ann):
    model.fit(log_2items_per_user)
    recs1 = model.predict(log_2items_per_user, k=2, filter_seen_items=False)

    model_with_ann.fit(log_2items_per_user)
    recs2 = model_with_ann.predict(
        log_2items_per_user, k=2, filter_seen_items=False
    )

    recs1 = recs1.toPandas().sort_values(
        ["user_idx", "item_idx"], ascending=False
    )
    recs2 = recs2.toPandas().sort_values(
        ["user_idx", "item_idx"], ascending=False
    )
    assert all(recs1.user_idx.values == recs2.user_idx.values)
    assert all(recs1.item_idx.values == recs2.item_idx.values)
