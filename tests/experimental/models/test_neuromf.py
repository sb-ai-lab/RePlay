# pylint: disable-all
from datetime import datetime

import pytest
import torch
import numpy as np
from pyspark.sql import functions as sf

from replay.data import INTERACTIONS_SCHEMA
from replay.experimental.models import NeuroMF
from replay.experimental.models.neuromf import NMF
from replay.experimental.models.base_rec import HybridRecommender, UserRecommender
from tests.utils import (
    del_files_by_pattern,
    find_file_by_pattern,
    spark,
    log,
    log_to_pred,
    long_log_with_features,
    user_features,
    sparkDataFrameEqual,
)
from replay.utils.model_handler import save, load


SEED = 123


def fit_predict_selected(model, train_log, inf_log, user_features, users):
    kwargs = {}
    if isinstance(model, (HybridRecommender, UserRecommender)):
        kwargs = {"user_features": user_features}
    model.fit(train_log, **kwargs)
    return model.predict(log=inf_log, users=users, k=1, **kwargs)


@pytest.fixture(scope="session", autouse=True)
def fix_seeds():
    torch.manual_seed(7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)


@pytest.fixture
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            (0, 0, date, 1.0),
            (0, 1, date, 1.0),
            (0, 2, date, 1.0),
            (1, 0, date, 1.0),
            (1, 1, date, 1.0),
            (0, 0, date, 1.0),
            (0, 1, date, 1.0),
            (0, 2, date, 1.0),
            (1, 0, date, 1.0),
            (1, 1, date, 1.0),
            (0, 0, date, 1.0),
            (0, 1, date, 1.0),
            (0, 2, date, 1.0),
            (1, 0, date, 1.0),
            (1, 1, date, 1.0),
            (2, 3, date, 1.0),
        ],
        schema=INTERACTIONS_SCHEMA,
    )


@pytest.fixture
def model():
    params = {
        "learning_rate": 0.5,
        "epochs": 1,
        "embedding_gmf_dim": 2,
        "embedding_mlp_dim": 2,
        "hidden_mlp_dims": [2],
    }
    model = NeuroMF(**params)
    return model


@pytest.mark.experimental
def test_equal_preds(long_log_with_features, tmp_path):
    path = (tmp_path / "test").resolve()
    model = NeuroMF()
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    loaded_model = load(path, NeuroMF)
    new_pred = loaded_model.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.experimental
def test_fit(log, model):
    model.fit(log)
    assert len(list(model.model.parameters())) == 12
    param_shapes = [
        (3, 2),
        (4, 2),
        (4, 1),
        (3, 1),
        (3, 2),
        (4, 2),
        (4, 1),
        (3, 1),
        (2, 4),
        (2,),
        (1, 4),
        (1,),
    ]
    for i, parameter in enumerate(model.model.parameters()):
        assert param_shapes[i] == tuple(parameter.shape)


@pytest.mark.experimental
def test_predict(log, model):
    model.fit(log)
    try:
        pred = model.predict(log=log, k=1)
        pred.count()
    except RuntimeError:  # noqa
        pytest.fail()


@pytest.mark.experimental
def test_check_gmf_only(log):
    params = {"learning_rate": 0.5, "epochs": 1, "embedding_gmf_dim": 2}
    model = NeuroMF(**params)
    try:
        model.fit(log)
    except RuntimeError:  # noqa
        pytest.fail()


@pytest.mark.experimental
def test_check_mlp_only(log):
    params = {
        "learning_rate": 0.5,
        "epochs": 1,
        "embedding_mlp_dim": 2,
        "hidden_mlp_dims": [2],
    }
    model = NeuroMF(**params)
    try:
        model.fit(log)
    except RuntimeError:  # noqa
        pytest.fail()


@pytest.mark.experimental
def test_check_simple_mlp_only(log):
    params = {"learning_rate": 0.5, "epochs": 1, "embedding_mlp_dim": 2}
    model = NeuroMF(**params)
    try:
        model.fit(log)
    except RuntimeError:  # noqa
        pytest.fail()


@pytest.mark.experimental
def test_embeddings_size():
    model = NeuroMF()
    assert model.embedding_gmf_dim == 128 and model.embedding_mlp_dim == 128

    model = NeuroMF(embedding_gmf_dim=16)
    assert model.embedding_gmf_dim == 16 and model.embedding_mlp_dim is None

    model = NeuroMF(embedding_gmf_dim=16, embedding_mlp_dim=32)
    assert model.embedding_gmf_dim == 16 and model.embedding_mlp_dim == 32


@pytest.mark.experimental
def test_negative_dims_exception():
    with pytest.raises(ValueError):
        NeuroMF(embedding_gmf_dim=-2, embedding_mlp_dim=-1)


@pytest.mark.experimental
def test_predict_pairs_warm_items_only(log, log_to_pred):
    model = NeuroMF()
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
    model = NeuroMF()
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


@pytest.mark.experimental
def test_predict_empty_log(log):
    model = NeuroMF()
    model.fit(log)
    model.predict(log.limit(0), 1)


@pytest.mark.experimental
def test_predict_cold_and_new_filter_out(long_log_with_features):
    model = NeuroMF()
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
