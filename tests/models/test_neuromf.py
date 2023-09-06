# pylint: disable-all
from datetime import datetime

import pytest
import torch
import numpy as np

from replay.data import LOG_SCHEMA
from replay.models import NeuroMF
from replay.models.neuromf import NMF
from tests.utils import del_files_by_pattern, find_file_by_pattern, spark


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
        schema=LOG_SCHEMA,
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


def test_predict(log, model):
    model.fit(log)
    try:
        pred = model.predict(log=log, k=1)
        pred.count()
    except RuntimeError:  # noqa
        pytest.fail()


def test_check_gmf_only(log):
    params = {"learning_rate": 0.5, "epochs": 1, "embedding_gmf_dim": 2}
    model = NeuroMF(**params)
    try:
        model.fit(log)
    except RuntimeError:  # noqa
        pytest.fail()


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


def test_check_simple_mlp_only(log):
    params = {"learning_rate": 0.5, "epochs": 1, "embedding_mlp_dim": 2}
    model = NeuroMF(**params)
    try:
        model.fit(log)
    except RuntimeError:  # noqa
        pytest.fail()


def test_save_load(log, model, spark):
    spark_local_dir = spark.conf.get("spark.local.dir")
    pattern = "best_neuromf_1_loss=\\d\\.\\d+.pt.?"
    del_files_by_pattern(spark_local_dir, pattern)

    model.fit(log=log)
    old_params = [
        param.detach().cpu().numpy() for param in model.model.parameters()
    ]
    path = find_file_by_pattern(spark_local_dir, pattern)
    assert path is not None

    new_model = NeuroMF(embedding_mlp_dim=1)
    new_model.model = NMF(3, 4, 2, 2, [2])
    assert len(old_params) == len(list(new_model.model.parameters()))

    new_model.load_model(path)
    for i, parameter in enumerate(new_model.model.parameters()):
        assert np.allclose(
            parameter.detach().cpu().numpy(), old_params[i], atol=1.0e-3,
        )


def test_embeddings_size():
    model = NeuroMF()
    assert model.embedding_gmf_dim == 128 and model.embedding_mlp_dim == 128

    model = NeuroMF(embedding_gmf_dim=16)
    assert model.embedding_gmf_dim == 16 and model.embedding_mlp_dim is None

    model = NeuroMF(embedding_gmf_dim=16, embedding_mlp_dim=32)
    assert model.embedding_gmf_dim == 16 and model.embedding_mlp_dim == 32


def test_negative_dims_exception():
    with pytest.raises(ValueError):
        NeuroMF(embedding_gmf_dim=-2, embedding_mlp_dim=-1)
