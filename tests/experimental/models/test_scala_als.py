import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

from pyspark.sql import functions as sf

from replay.experimental.models import ScalaALSWrap as ALSWrap
from replay.experimental.models.base_rec import HybridRecommender, UserRecommender
from replay.experimental.scenarios.two_stages.two_stages_scenario import get_first_level_model_features
from replay.experimental.utils.model_handler import save
from replay.models.extensions.ann.entities.hnswlib_param import HnswlibParam
from replay.models.extensions.ann.index_builders.executor_hnswlib_index_builder import ExecutorHnswlibIndexBuilder
from replay.models.extensions.ann.index_stores.shared_disk_index_store import SharedDiskIndexStore
from replay.utils.model_handler import load
from tests.utils import sparkDataFrameEqual

SEED = 123


def fit_predict_selected(model, train_log, inf_log, user_features, users):
    kwargs = {}
    if isinstance(model, (HybridRecommender, UserRecommender)):
        kwargs = {"user_features": user_features}
    model.fit(train_log, **kwargs)
    return model.predict(log=inf_log, users=users, k=1, **kwargs)


@pytest.fixture
def model():
    model = ALSWrap(2, implicit_prefs=False)
    model._seed = 42
    return model


@pytest.fixture
def model_with_ann(tmp_path):
    model = ALSWrap(
        rank=2,
        implicit_prefs=False,
        seed=42,
        index_builder=ExecutorHnswlibIndexBuilder(
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


@pytest.mark.experimental
def test_equal_preds(long_log_with_features, tmp_path):
    path = (tmp_path / "test").resolve()
    model = ALSWrap()
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    loaded_model = load(path, ALSWrap)
    new_pred = loaded_model.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.experimental
def test_works(log, model):
    pred = model.fit_predict(log, k=1)
    assert pred.count() == 4


@pytest.mark.experimental
def test_diff_feedback_type(log, model):
    pred_exp = model.fit_predict(log, k=1)
    model.implicit_prefs = True
    pred_imp = model.fit_predict(log, k=1)
    assert not np.allclose(
        pred_exp.toPandas().sort_values("user_idx")["relevance"].values,
        pred_imp.toPandas().sort_values("user_idx")["relevance"].values,
    )


@pytest.mark.experimental
def test_enrich_with_features(log, model):
    model.fit(log.filter(sf.col("user_idx").isin([0, 2])))
    res = get_first_level_model_features(model, log.filter(sf.col("user_idx").isin([0, 1])))

    cold_user_and_item = res.filter((sf.col("user_idx") == 1) & (sf.col("item_idx") == 3))
    row_dict = cold_user_and_item.collect()[0].asDict()
    assert row_dict["_if_0"] == row_dict["_uf_0"] == row_dict["_fm_1"] == 0.0

    warm_user_and_item = res.filter((sf.col("user_idx") == 0) & (sf.col("item_idx") == 0))
    row_dict = warm_user_and_item.collect()[0].asDict()
    np.allclose(
        [row_dict["_fm_1"], row_dict["_if_1"] * row_dict["_uf_1"]],
        [4.093189725967505, row_dict["_fm_1"]],
    )

    cold_user_warm_item = res.filter((sf.col("user_idx") == 1) & (sf.col("item_idx") == 0))
    row_dict = cold_user_warm_item.collect()[0].asDict()
    np.allclose(
        [row_dict["_if_1"], row_dict["_if_1"] * row_dict["_uf_1"]],
        [-2.938199281692505, 0],
    )


@pytest.mark.experimental
@pytest.mark.parametrize("filter_seen_items", [True, False])
def test_ann_predict(log, model, model_with_ann, filter_seen_items):
    model.fit(log)
    recs1 = model.predict(log, k=1, filter_seen_items=filter_seen_items)

    model_with_ann.fit(log)
    recs2 = model_with_ann.predict(log, k=1, filter_seen_items=filter_seen_items)

    recs1 = recs1.toPandas().sort_values(["user_idx", "item_idx"], ascending=False)
    recs2 = recs2.toPandas().sort_values(["user_idx", "item_idx"], ascending=False)
    assert recs1.user_idx.equals(recs2.user_idx)
    assert recs1.item_idx.equals(recs2.item_idx)


@pytest.mark.experimental
def test_predict_pairs_warm_items_only(log, log_to_pred):
    model = ALSWrap(seed=SEED)
    model.fit(log)
    recs = model.predict(
        log.unionByName(log_to_pred),
        k=3,
        users=log_to_pred.select("user_idx").distinct(),
        items=log_to_pred.select("item_idx").distinct(),
        filter_seen_items=False,
    )

    pairs_pred = model.predict_pairs(
        pairs=log_to_pred.select("user_idx", "item_idx"),
        log=log.unionByName(log_to_pred),
    )

    condition = ~sf.col("item_idx").isin([4, 5])
    if not model.can_predict_cold_users:
        condition = condition & (sf.col("user_idx") != 4)

    sparkDataFrameEqual(
        pairs_pred.select("user_idx", "item_idx"),
        log_to_pred.filter(condition).select("user_idx", "item_idx"),
    )

    recs_joined = (
        pairs_pred.withColumnRenamed("relevance", "pairs_relevance")
        .join(recs, on=["user_idx", "item_idx"], how="left")
        .sort("user_idx", "item_idx")
    )

    assert np.allclose(
        recs_joined.select("relevance").toPandas().to_numpy(),
        recs_joined.select("pairs_relevance").toPandas().to_numpy(),
    )


@pytest.mark.experimental
def test_predict_pairs_k(log):
    model = ALSWrap(seed=SEED)
    model.fit(log)

    pairs_pred_k = model.predict_pairs(
        pairs=log.select("user_idx", "item_idx"),
        log=log,
        k=1,
    )

    pairs_pred = model.predict_pairs(
        pairs=log.select("user_idx", "item_idx"),
        log=log,
        k=None,
    )

    assert pairs_pred_k.groupBy("user_idx").count().filter(sf.col("count") > 1).count() == 0

    assert pairs_pred.groupBy("user_idx").count().filter(sf.col("count") > 1).count() > 0


@pytest.mark.experimental
def test_predict_empty_log(log):
    model = ALSWrap(seed=SEED)
    model.fit(log)
    model.predict(log.limit(0), 1)


@pytest.mark.experimental
def test_predict_pairs_raises_pairs_format(log):
    model = ALSWrap(seed=SEED)
    with pytest.raises(ValueError, match="pairs must be a dataframe with .*"):
        model.fit(log)
        model.predict_pairs(log, log)


@pytest.mark.experimental
@pytest.mark.parametrize(
    "als_model, metric",
    [
        (ALSWrap(seed=SEED), "euclidean_distance_sim"),
        (ALSWrap(seed=SEED), "dot_product"),
        (ALSWrap(seed=SEED), "cosine_similarity"),
    ],
    ids=[
        "als_euclidean",
        "als_dot",
        "als_cosine",
    ],
)
def test_get_nearest_items(log, als_model, metric):
    als_model.fit(log.filter(sf.col("item_idx") != 3))
    res = als_model.get_nearest_items(items=[0, 1], k=2, metric=metric)

    assert res.count() == 4
    assert set(res.toPandas().to_dict()["item_idx"].values()) == {
        0,
        1,
    }

    res = als_model.get_nearest_items(items=[0, 1], k=1, metric=metric)
    assert res.count() == 2

    # filter neighbours
    res = als_model.get_nearest_items(
        items=[0, 1],
        k=4,
        metric=metric,
        candidates=[0, 3],
    )
    assert res.count() == 1
    assert len(set(res.toPandas().to_dict()["item_idx"].values()).difference({0, 1})) == 0


@pytest.mark.experimental
@pytest.mark.parametrize("metric", ["absent", None])
def test_nearest_items_raises(log, metric):
    model = ALSWrap()
    model.fit(log)
    with pytest.raises(ValueError, match=r"Select one of the valid distance metrics.*"):
        model.get_nearest_items(items=[0, 1], k=2, metric=metric)


@pytest.mark.experimental
def test_predict_cold_and_new_filter_out(long_log_with_features):
    model = ALSWrap(rank=2, seed=SEED)
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features,
        user_features=None,
        users=[0, 3],
    )
    # assert new/cold users are filtered out in `predict`
    if not model.can_predict_cold_users:
        assert pred.count() == 0
    else:
        assert 1 <= pred.count() <= 2


@pytest.mark.experimental
def test_predict_pairs_to_file(spark, long_log_with_features, tmp_path):
    model = ALSWrap(rank=2, seed=SEED)
    path = str((tmp_path / "pred.parquet").resolve().absolute())
    model.fit(long_log_with_features)
    model.predict_pairs(
        log=long_log_with_features,
        pairs=long_log_with_features.filter(sf.col("user_idx") == 1).select("user_idx", "item_idx"),
        recs_file_path=path,
    )
    pred_cached = model.predict_pairs(
        log=long_log_with_features,
        pairs=long_log_with_features.filter(sf.col("user_idx") == 1).select("user_idx", "item_idx"),
        recs_file_path=None,
    )
    pred_from_file = spark.read.parquet(path)
    sparkDataFrameEqual(pred_cached, pred_from_file)


@pytest.mark.experimental
def test_predict_to_file(spark, long_log_with_features, tmp_path):
    model = ALSWrap(rank=2, seed=SEED)
    path = str((tmp_path / "pred.parquet").resolve().absolute())
    model.fit_predict(long_log_with_features, k=10, recs_file_path=path)
    pred_cached = model.predict(long_log_with_features, k=10, recs_file_path=None)
    pred_from_file = spark.read.parquet(path)
    sparkDataFrameEqual(pred_cached, pred_from_file)
