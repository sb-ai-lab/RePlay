# pylint: disable-all
from datetime import datetime

import pytest
import numpy as np
import torch
from pyspark.sql import functions as sf

from replay.constants import LOG_SCHEMA
from replay.models import MultVAE
from replay.models.mult_vae import VAE
from tests.test_utils import del_files_by_pattern, find_file_by_pattern
from tests.utils import spark, sparkDataFrameEqual


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
            ("0", "0", date, 1.0),
            ("0", "2", date, 1.0),
            ("1", "0", date, 1.0),
            ("1", "1", date, 1.0),
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def other_log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            ("2", "0", date, 1.0),
            ("2", "1", date, 1.0),
            ("0", "0", date, 1.0),
            ("0", "2", date, 1.0),
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def model():
    params = {
        "learning_rate": 0.5,
        "epochs": 1,
        "latent_dim": 1,
        "hidden_dim": 1,
    }
    return MultVAE(**params)


def test_fit(log, model):
    model.fit(log)
    param_shapes = [
        (1, 3),
        (1,),
        (2, 1),
        (2,),
        (1, 1),
        (1,),
        (3, 1),
        (3,),
    ]
    assert len(list(model.model.parameters())) == len(param_shapes)
    for i, parameter in enumerate(model.model.parameters()):
        assert param_shapes[i] == tuple(parameter.shape)


def test_predict(log, other_log, model):
    model.fit(log)
    recs = model.predict(other_log, k=1)
    assert np.allclose(
        recs.toPandas()
        .sort_values("user_id")[["user_id", "item_id"]]
        .astype(int)
        .values,
        [[0, 1], [2, 2]],
        atol=1.0e-3,
    )


def test_predict_pairs(log, other_log, model):
    model.fit(log)
    recs = model.predict(other_log, k=3, filter_seen_items=False, users=["2"])
    pairs_pred = model.predict_pairs(
        pairs=other_log.select("user_id", "item_id").filter(
            sf.col("user_id") == "2"
        ),
        log=other_log,
    )
    sparkDataFrameEqual(
        pairs_pred.select("user_id", "item_id"),
        other_log.select("user_id", "item_id").filter(
            sf.col("user_id") == "2"
        ),
    )
    recs.show()
    pairs_pred.show()
    recs_joined = (
        pairs_pred.withColumnRenamed("relevance", "pairs_relevance")
        .join(recs, on=["user_id", "item_id"], how="left")
        .sort("user_id", "item_id")
    )
    recs_joined.show()
    assert np.allclose(
        recs_joined.select("relevance").toPandas().to_numpy(),
        recs_joined.select("pairs_relevance").toPandas().to_numpy(),
    )


def test_save_load(log, model, spark):
    spark_local_dir = spark.conf.get("spark.local.dir")
    pattern = "best_multvae_1_loss=-\\d\\.\\d+.pth"
    del_files_by_pattern(spark_local_dir, pattern)

    model.fit(log=log)
    old_params = [
        param.detach().cpu().numpy() for param in model.model.parameters()
    ]
    path = find_file_by_pattern(spark_local_dir, pattern)
    assert path is not None

    new_model = MultVAE()
    new_model.model = VAE(item_count=3, latent_dim=1, hidden_dim=1)
    assert len(old_params) == len(list(new_model.model.parameters()))

    new_model.load_model(path)
    for i, parameter in enumerate(new_model.model.parameters()):
        assert np.allclose(
            parameter.detach().cpu().numpy(), old_params[i], atol=1.0e-3,
        )
