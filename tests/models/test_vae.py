# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

import pytest
import numpy as np
import torch
import pyspark.sql.functions as sf

from replay.models import MultVAE
from replay.models.mult_vae import VAE
from tests.utils import (
    del_files_by_pattern,
    find_file_by_pattern,
    log,
    log2,
    spark,
)


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


def test_predict(log, model):
    recs = model.predict(log, users=[0, 1, 7], k=1)
    # new users with history
    assert recs.filter(sf.col("user_idx") == 0).count() == 1
    # cold user
    assert recs.filter(sf.col("user_idx") == 7).count() == 0
    assert recs.count() == 2


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


def test_save_load(log, model, spark):
    spark_local_dir = spark.conf.get("spark.local.dir")
    pattern = "best_multvae_1_loss=-\\d\\.\\d+.pt.?"
    del_files_by_pattern(spark_local_dir, pattern)

    model.fit(log=log)
    old_params = [
        param.detach().cpu().numpy() for param in model.model.parameters()
    ]
    path = find_file_by_pattern(spark_local_dir, pattern)
    assert path is not None

    new_model = MultVAE()
    new_model.model = VAE(item_count=4, latent_dim=1, hidden_dim=1)
    assert len(old_params) == len(list(new_model.model.parameters()))

    new_model.load_model(path)
    for i, parameter in enumerate(new_model.model.parameters()):
        assert np.allclose(
            parameter.detach().cpu().numpy(), old_params[i], atol=1.0e-3,
        )
