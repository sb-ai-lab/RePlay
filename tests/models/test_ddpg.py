# pylint: disable-all
from datetime import datetime

import pytest
import torch
import numpy as np

from replay.constants import LOG_SCHEMA
from replay.models import DDPG
from replay.models.ddpg import ActorDRR, Env, to_np
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
    model = DDPG()
    model.fit(log)
    return model


def test_fit(log, model):
    model.fit(log)
    assert len(list(model.model.parameters())) == 10
    param_shapes = [
        (16, 24),
        (16,),
        (16,),
        (16,),
        (8, 16),
        (8,),
        (5000, 8),
        (200001, 8),
        (1, 5, 1),
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
    spark_local_dir = "./logs/tmp/"
    pattern = "model_final.pt"
    del_files_by_pattern(spark_local_dir, pattern)

    model.fit(log=log)
    old_params = [
        param.detach().cpu().numpy() for param in model.model.parameters()
    ]
    path = find_file_by_pattern(spark_local_dir, pattern)
    assert path is not None

    new_model = DDPG()
    new_model.model = ActorDRR(
        user_num=5000,
        item_num=200000,
        embedding_dim=8,
        hidden_dim=16,
        memory_size=5,
    )
    assert len(old_params) == len(list(new_model.model.parameters()))

    new_model.load_model(path)
    for i, parameter in enumerate(new_model.model.parameters()):
        assert np.allclose(
            parameter.detach().cpu().numpy(),
            old_params[i],
            atol=1.0e-3,
        )


def test_env_step(log, model, user=0):
    replay_buffer = ReplayBuffer()
    train_matrix, _, _, item_num, _ = model._preprocess_log(log)
    model.model.environment.update_env(matrix=train_matrix, item_count=item_num)

    user, memory = model.model.environment.reset(user)

    action_emb = model.model(user, memory)
    action = model.model.get_action(
        action_emb, model.model.environment.available_items,
    )

    model.model.environment.memory[to_np(user), to_np(action)] = 1

    user, new_memory, reward, _ = model.model.environment.step(
        action, action_emb, replay_buffer
    )
    assert new_memory[user][0][-1] == action
