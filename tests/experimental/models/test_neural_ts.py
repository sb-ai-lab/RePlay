# pylint: disable-all
import os
from datetime import datetime

import pytest

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

from pyspark.sql import functions as sf

from replay.data import get_schema
from replay.experimental.models import NeuralTS
from tests.utils import sparkDataFrameEqual

SEED = 123


@pytest.fixture
def log(spark):
    date1 = datetime(2019, 1, 1)
    date2 = datetime(2019, 1, 2)
    date3 = datetime(2019, 1, 3)
    return spark.createDataFrame(
        data=[
            [0, 0, date1, 1.0],
            [1, 0, date1, 1.0],
            [2, 1, date2, 2.0],
            [2, 1, date2, 2.0],
            [1, 1, date3, 2.0],
            [2, 2, date3, 2.0],
            [0, 2, date3, 2.0],
        ],
        schema=get_schema("user_idx", "item_idx", "timestamp", "relevance"),
    )


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame([(0, 2.0, 5.0), (1, 0.0, -5.0), (2, 4.0, 3.0)]).toDF(
        "user_idx", "user_feature_cat", "user_feature_cont"
    )


@pytest.fixture
def item_features(spark):
    return spark.createDataFrame([(0, 4.0, 5.0), (1, 5.0, 4.0), (2, 1.5, 3.3)]).toDF(
        "item_idx", "item_feature_cat", "item_feature_cont"
    )


@pytest.fixture
def model():
    cols_item = {"continuous_cols": [], "cat_embed_cols": [], "wide_cols": []}
    cols_user = {"continuous_cols": [], "cat_embed_cols": [], "wide_cols": []}

    model = NeuralTS(
        user_cols=cols_user,
        item_cols=cols_item,
        dim_head=1,
        deep_out_dim=1,
        hidden_layers=[2, 5],
        embedding_sizes=[2, 2, 4],
        wide_out_dim=1,
        head_dropout=0.8,
        deep_dropout=0.4,
        n_epochs=1,
        opt_lr=3e-4,
        lr_min=1e-5,
        use_gpu=False,
        plot_dir=None,
        is_warp_loss=True,
        cnt_neg_samples=1,
        cnt_samples_for_predict=2,
        eps=1.0,
    )

    return model


@pytest.fixture
def model_with_features():
    cols_item = {
        "continuous_cols": ["item_feature_cont"],
        "cat_embed_cols": ["item_feature_cat"],
        "wide_cols": ["item_feature_cat", "item_feature_cont"],
    }

    cols_user = {
        "continuous_cols": ["user_feature_cont"],
        "cat_embed_cols": ["user_feature_cat"],
        "wide_cols": ["user_feature_cat", "user_feature_cont"],
    }

    model_with_features = NeuralTS(
        user_cols=cols_user,
        item_cols=cols_item,
        dim_head=1,
        deep_out_dim=1,
        hidden_layers=[2, 5],
        embedding_sizes=[2, 2, 4],
        wide_out_dim=1,
        head_dropout=0.8,
        deep_dropout=0.4,
        n_epochs=2,
        opt_lr=3e-4,
        lr_min=1e-5,
        use_gpu=False,
        plot_dir="plot.png",
        is_warp_loss=False,
        cnt_neg_samples=1,
        cnt_samples_for_predict=2,
        eps=1.0,
    )

    return model_with_features


@pytest.mark.experimental
def test_equal_preds(model, user_features, item_features, log):
    dir_name = "test"
    model.fit(log, user_features=user_features, item_features=item_features)
    torch.manual_seed(SEED)
    base_pred = model.predict(log, 5, user_features=user_features, item_features=item_features)
    model.model_save(dir_name)
    model.model_load(dir_name)
    torch.manual_seed(SEED)
    new_pred = model.predict(log, 5, user_features=user_features, item_features=item_features)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.experimental
def test_use_features(model_with_features, user_features, item_features, log):
    model_with_features.fit(log, user_features=user_features, item_features=item_features)

    users = user_features.select("user_idx").distinct()
    items = item_features.select("item_idx").distinct()

    base_pred = model_with_features._predict(
        log, 5, users=users, items=items, user_features=user_features, item_features=item_features
    )

    n_users = users.count()
    n_items = items.count()

    assert base_pred.count() == n_users * n_items
    assert os.path.exists("plot.png")


@pytest.mark.experimental
def test_predict_pairs(log, user_features, item_features, model):
    model.fit(log, user_features, item_features)

    pred = model.predict_pairs(
        log.filter(sf.col("user_idx") == 1).select("user_idx", "item_idx"),
        user_features=user_features,
        item_features=item_features,
    )
    assert pred.count() == 2
    assert pred.select("user_idx").distinct().collect()[0][0] == 1


@pytest.mark.experimental
def test_predict_empty_log(model_with_features, user_features, item_features, log):
    model_with_features.fit(log, user_features=user_features, item_features=item_features)
    users = user_features.select("user_idx").distinct()
    items = item_features.select("item_idx").distinct()

    model_with_features._predict(
        log.limit(0), 1, users=users, items=items, user_features=user_features, item_features=item_features
    )
