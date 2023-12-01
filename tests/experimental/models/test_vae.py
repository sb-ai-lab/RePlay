# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

import pyspark.sql.functions as sf

from replay.experimental.models import MultVAE
from replay.experimental.models.base_rec import HybridRecommender, UserRecommender
from replay.utils.model_handler import load, save
from tests.utils import (
    del_files_by_pattern,
    find_file_by_pattern,
    log,
    log2,
    log_to_pred,
    long_log_with_features,
    spark,
    sparkDataFrameEqual,
    user_features,
)

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
def model(log):
    params = {
        "learning_rate": 0.5,
        "epochs": 1,
        "latent_dim": 1,
        "hidden_dim": 1,
    }
    model = MultVAE(**params)
    model.fit(log.filter(sf.col("user_idx") != 0))
    return model


@pytest.mark.experimental
def test_equal_preds(long_log_with_features, tmp_path):
    path = (tmp_path / "test").resolve()
    model = MultVAE()
    model.fit(long_log_with_features)
    base_pred = model.predict(long_log_with_features, 5)
    save(model, path)
    loaded_model = load(path, MultVAE)
    new_pred = loaded_model.predict(long_log_with_features, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.experimental
def test_fit(model):
    param_shapes = [
        (1, 4),
        (1,),
        (2, 1),
        (2,),
        (1, 1),
        (1,),
        (4, 1),
        (4,),
    ]
    assert len(list(model.model.parameters())) == len(param_shapes)
    for i, parameter in enumerate(model.model.parameters()):
        assert param_shapes[i] == tuple(parameter.shape)


@pytest.mark.experimental
def test_predict(log, model):
    recs = model.predict(log, users=[0, 1, 7], k=1)
    # new users with history
    assert recs.filter(sf.col("user_idx") == 0).count() == 1
    # cold user
    assert recs.filter(sf.col("user_idx") == 7).count() == 0
    assert recs.count() == 2


@pytest.mark.experimental
def test_predict_pairs(log, log2, model):
    recs = model.predict_pairs(
        pairs=log2.select("user_idx", "item_idx"), log=log
    )
    assert (
        recs.count()
        == (
            log2.join(
                log.select("user_idx").distinct(), on="user_idx", how="inner"
            ).join(
                log.select("item_idx").distinct(), on="item_idx", how="inner"
            )
        ).count()
    )


@pytest.mark.experimental
def test_predict_pairs_warm_items_only(log, log_to_pred):
    model = MultVAE()
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
    model = MultVAE()
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
    model = MultVAE()
    model.fit(log)
    model.predict(log.limit(0), 1)


@pytest.mark.experimental
def test_predict_new_users(long_log_with_features, user_features):
    model = MultVAE()
    pred = fit_predict_selected(
        model,
        train_log=long_log_with_features.filter(sf.col("user_idx") != 0),
        inf_log=long_log_with_features,
        user_features=user_features.drop("gender"),
        users=[0],
    )
    assert pred.count() == 1
    assert pred.collect()[0][0] == 0


@pytest.mark.experimental
def test_predict_cold_and_new_filter_out(long_log_with_features):
    model = MultVAE()
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
