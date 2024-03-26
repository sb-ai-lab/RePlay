from datetime import datetime

import numpy as np
import pytest

from replay.data import get_schema
from replay.models import Word2VecRec
from replay.models.extensions.ann.entities.hnswlib_param import HnswlibParam
from replay.models.extensions.ann.index_builders.driver_hnswlib_index_builder import DriverHnswlibIndexBuilder
from replay.models.extensions.ann.index_stores.shared_disk_index_store import SharedDiskIndexStore
from tests.utils import (
    create_dataset,
)

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import functions as sf

from replay.utils.spark_utils import vector_dot

INTERACTIONS_SCHEMA = get_schema("user_idx", "item_idx", "timestamp", "relevance")


@pytest.fixture
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            [0, 0, date, 1.0],
            [1, 0, date, 1.0],
            [2, 1, date, 2.0],
            [2, 1, date, 2.0],
            [1, 1, date, 2.0],
            [2, 3, date, 2.0],
            [0, 3, date, 2.0],
        ],
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture
def model():
    return Word2VecRec(rank=1, window_size=1, use_idf=True, seed=42, min_count=0)


@pytest.fixture
def model_with_ann(tmp_path):
    model = Word2VecRec(
        rank=1,
        window_size=1,
        use_idf=True,
        seed=42,
        min_count=0,
        index_builder=DriverHnswlibIndexBuilder(
            index_params=HnswlibParam(
                space="ip",
                m=100,
                ef_c=2000,
                post=0,
                ef_s=2000,
            ),
            index_store=SharedDiskIndexStore(warehouse_dir=str(tmp_path), index_dir="hnswlib_index"),
        ),
    )
    return model


@pytest.mark.spark
def test_fit(log, model):
    dataset = create_dataset(log)
    model.fit(dataset)
    vectors = (
        model.vectors.select(
            "item",
            vector_dot(sf.col("vector"), sf.col("vector")).alias("norm"),
        )
        .toPandas()
        .to_numpy()
    )
    assert np.allclose(
        vectors,
        [[1, 5.33072205e-04], [0, 1.54904364e-01], [3, 2.13002899e-01]],
        atol=1e-04,
    )


@pytest.mark.spark
def test_predict(log, model):
    dataset = create_dataset(log)
    model.fit(dataset)
    recs = model.predict(dataset, k=1)
    recs.show()
    assert np.allclose(
        recs.toPandas().sort_values("user_idx").relevance,
        [1.0003180271011836, 0.9653348251181987, 0.972993367280087],
    )


# here we use `test.utils.log` because we can't build the hnsw index on `log` data
@pytest.mark.spark
def test_word2vec_predict_filter_seen_items(log2, model, model_with_ann):
    dataset = create_dataset(log2)
    model.fit(dataset)
    recs1 = model.predict(dataset, k=1)

    model_with_ann.fit(dataset)
    recs2 = model_with_ann.predict(dataset, k=1)

    recs1 = recs1.toPandas().sort_values(["user_idx", "item_idx"], ascending=False)
    recs2 = recs2.toPandas().sort_values(["user_idx", "item_idx"], ascending=False)
    assert recs1.user_idx.equals(recs2.user_idx)
    assert recs1.item_idx.equals(recs2.item_idx)


@pytest.mark.spark
def test_word2vec_predict(log2, model, model_with_ann):
    dataset = create_dataset(log2)
    model.fit(dataset)
    recs1 = model.predict(dataset, k=2, filter_seen_items=False)

    model_with_ann.fit(dataset)
    recs2 = model_with_ann.predict(dataset, k=2, filter_seen_items=False)

    recs1 = recs1.toPandas().sort_values(["user_idx", "item_idx"], ascending=False)
    recs2 = recs2.toPandas().sort_values(["user_idx", "item_idx"], ascending=False)
    assert recs1.user_idx.equals(recs2.user_idx)
    assert recs1.item_idx.equals(recs2.item_idx)
