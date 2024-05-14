from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

from replay.experimental.models.dt4rec.dt4rec import DT4Rec
from replay.experimental.models.dt4rec.gpt1 import GPT, Block, CausalSelfAttention
from replay.experimental.models.dt4rec.utils import (
    ValidateDataset,
    create_dataset as create_dt4rec_dataset,
    fast_create_dataset,
    matrix2df,
)
from replay.experimental.preprocessing.data_preparator import DataPreparator, Indexer


@dataclass
class TestConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    n_layer = 1
    n_head = 1
    memory_size = 3
    n_embd = 8
    block_size = 12
    max_timestep = 10

    item_num = 10
    user_num = 10

    vocab_size = item_num + 1


@pytest.mark.experimental
@pytest.mark.torch
def test_casual_self_attention():
    cfg = TestConfig()
    csa = CausalSelfAttention(cfg)
    batch = torch.randn(3, 4, cfg.n_embd)
    assert csa(batch).shape == batch.shape


@pytest.mark.experimental
@pytest.mark.torch
def test_block():
    cfg = TestConfig()
    block = Block(cfg)
    batch = torch.randn(3, 4, cfg.n_embd)
    assert block(batch).shape == batch.shape


@pytest.mark.experimental
@pytest.mark.torch
def test_gpt1():
    cfg = TestConfig()
    gpt = GPT(cfg)
    batch_size = 4
    states = torch.randint(0, cfg.item_num, (batch_size, cfg.block_size // 3, 3))
    actions = torch.randint(0, cfg.item_num, (batch_size, cfg.block_size // 3, 1))
    rtgs = torch.randn(batch_size, cfg.block_size // 3, 1)
    timesteps = torch.randint(0, cfg.max_timestep, (batch_size, 1, 1))
    users = torch.randint(0, cfg.user_num, (batch_size, 1))

    assert gpt(states, actions, rtgs, timesteps, users).shape == (4, cfg.block_size // 3, cfg.vocab_size)


@pytest.mark.experimental
@pytest.mark.torch
def test_create_dataset_good_items():
    df = pd.DataFrame({"timestamp": [10, 11, 12], "user_idx": [0, 0, 0], "item_idx": [3, 4, 5], "relevance": [5, 5, 5]})
    ans = [
        {
            "states": np.array([[0, 0, 0], [0, 0, 3], [0, 3, 4], [3, 4, 5]]),
            "actions": np.array([3, 4, 5]),
            "rewards": np.array([1, 1, 1]),
            "rtgs": np.array([3, 2, 1]),
        }
    ]
    result = create_dt4rec_dataset(df, user_num=1, item_pad=0)
    for key, value in ans[0].items():
        assert (value == result[0][key]).all()


@pytest.mark.experimental
@pytest.mark.torch
def test_fast_create_dataset_good_items():
    df = pd.DataFrame({"timestamp": [10, 11, 12], "user_idx": [0, 0, 0], "item_idx": [3, 4, 5], "relevance": [5, 5, 5]})
    ans = [
        {
            "states": np.array([[0, 0, 0], [0, 0, 3], [0, 3, 4], [3, 4, 5]]),
            "actions": np.array([3, 4, 5]),
            "rewards": np.array([1, 1, 1]),
            "rtgs": np.array([3, 2, 1]),
        }
    ]
    result = fast_create_dataset(df, user_num=1, item_pad=0)
    for key, value in ans[0].items():
        assert (value == result[0][key]).all()


@pytest.mark.experimental
@pytest.mark.torch
def test_create_dataset_bad_items():
    df = pd.DataFrame({"timestamp": [10, 11, 12], "user_idx": [0, 0, 0], "item_idx": [3, 4, 5], "relevance": [0, 0, 0]})
    ans = [
        {
            "states": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "actions": np.array([3, 4, 5]),
            "rewards": np.array([0, 0, 0]),
            "rtgs": np.array([0, 0, 0]),
        }
    ]
    result = create_dt4rec_dataset(df, user_num=1, item_pad=0)

    for key, value in ans[0].items():
        assert (value == result[0][key]).all()

    val_ds = ValidateDataset(result, 10, [0], [3])
    val_ds[0]


@pytest.mark.experimental
@pytest.mark.torch
def test_fast_create_dataset_bag_items():
    df = pd.DataFrame({"timestamp": [10, 11, 12], "user_idx": [0, 0, 0], "item_idx": [3, 4, 5], "relevance": [0, 0, 0]})
    ans = [
        {
            "states": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "actions": np.array([3, 4, 5]),
            "rewards": np.array([0, 0, 0]),
            "rtgs": np.array([0, 0, 0]),
        }
    ]
    result = fast_create_dataset(df, user_num=1, item_pad=0)
    for key, value in ans[0].items():
        assert (value == result[0][key]).all()


@pytest.mark.experimental
@pytest.mark.torch
def test_matrix2df():
    matrix = torch.tensor([[1, 2], [3, 4]])
    df = matrix2df(matrix)
    assert (df[df.user_idx == 0].relevance.values == np.array([1, 3])).any()
    assert (df[df.item_idx == 0].relevance.values == np.array([1, 3])).all()


@pytest.mark.experimental
@pytest.mark.spark
@pytest.mark.torch
def test_train():
    df = pd.DataFrame(
        {
            "timestamp": list(range(60)) + list(range(60)),
            "user_id": [0 for i in range(60)] + [1 for i in range(60)],
            "item_id": list(range(60)) + list(range(60)),
            "rating": [1 for i in range(60)] + [1 for i in range(60)],
        }
    )

    preparator = DataPreparator()
    prepared_log = preparator.transform(
        columns_mapping={
            "user_id": "user_id",
            "item_id": "item_id",
            "relevance": "rating",
            "timestamp": "timestamp",
        },
        data=df,
    )

    indexer = Indexer(user_col="user_id", item_col="item_id")
    indexer.fit(users=prepared_log.select("user_id"), items=prepared_log.select("item_id"))
    log = indexer.transform(prepared_log)

    item_num = log.toPandas()["item_idx"].max() + 1
    user_num = log.toPandas()["user_idx"].max() + 1

    decision_transformer = DT4Rec(item_num, user_num, use_cuda=False)
    decision_transformer.train_batch_size = 10
    decision_transformer.val_batch_size = 10
    decision_transformer.fit(log)
    decision_transformer.predict(log=log, k=1)

    assert True
