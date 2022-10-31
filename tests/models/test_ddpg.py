# pylint: disable-all
from datetime import datetime

import pytest
import torch
import numpy as np
from pytorch_ranger import Ranger

from replay.constants import LOG_SCHEMA
from replay.models import DDPG
from replay.models.ddpg import ActorDRR, Env, ReplayBuffer, to_np
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
def model(log):
    model = DDPG(user_num=5, item_num=5)
    model.batch_size = 1
    return model


def test_fit(log, model, user_num=5, item_num=5):
    model.fit(log)
    assert len(list(model.model.parameters())) == 10
    param_shapes = [
        (16, 24),
        (16,),
        (16,),
        (16,),
        (8, 16),
        (8,),
        (user_num, 8),
        (item_num + 1, 8),
        (1, 5, 1),
        (1,),
    ]
    for i, parameter in enumerate(model.model.parameters()):
        assert param_shapes[i] == tuple(parameter.shape)


def test_predict(log, model):
    model.ou_noise.noise_type = "gauss"
    model.replay_buffer.capacity = 4
    model.batch_size = 4
    model.fit(log)
    try:
        pred = model.predict(log=log, k=1)
        pred.count()
    except RuntimeError:  # noqa
        pytest.fail()


def test_save_load(log, model, spark, user_num=5, item_num=5):
    spark_local_dir = "./logs/tmp/"
    pattern = "model_final.pt"
    del_files_by_pattern(spark_local_dir, pattern)

    model.fit(log=log)
    old_params = [
        param.detach().cpu().numpy() for param in model.model.parameters()
    ]
    path = find_file_by_pattern(spark_local_dir, pattern)
    assert path is not None

    new_model = DDPG(user_num=user_num, item_num=item_num)
    new_model.model = ActorDRR(
        user_num=user_num,
        item_num=item_num,
        embedding_dim=8,
        hidden_dim=16,
        memory_size=5,
    )
    assert len(old_params) == len(list(new_model.model.parameters()))

    new_model._load_model(path)
    for i, parameter in enumerate(new_model.model.parameters()):
        assert np.allclose(
            parameter.detach().cpu().numpy(), old_params[i], atol=1.0e-3,
        )

