# pylint: disable-all
from datetime import datetime

import pytest
import torch
import numpy as np
import pandas as pd

from replay.models.dt4rec.utils import create_dataset, fast_create_dataset
from replay.models.dt4rec.gpt1 import CausalSelfAttention, Block, GPT
from dataclasses import dataclass

from replay.metrics import MAP, MRR, NDCG, Coverage, HitRate, Surprisal
from replay.metrics.experiment import Experiment
from replay.models import DT4Rec
from replay.preprocessing.data_preparator import DataPreparator, Indexer
from replay.splitters import DateSplitter

from tests.utils import del_files_by_pattern, find_file_by_pattern, spark


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


def test_casual_self_attention():
    cfg = TestConfig()
    csa = CausalSelfAttention(cfg)
    batch = torch.randn(3, 4, cfg.n_embd)
    assert csa(batch).shape == batch.shape


def test_block():
    cfg = TestConfig()
    block = Block(cfg)
    batch = torch.randn(3, 4, cfg.n_embd)
    assert block(batch).shape == batch.shape


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
    result = create_dataset(df, user_num=1, item_pad=0)
    for key, value in ans[0].items():
        assert (value == result[0][key]).all()


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


def test_create_dataset_bag_items():
    df = pd.DataFrame({"timestamp": [10, 11, 12], "user_idx": [0, 0, 0], "item_idx": [3, 4, 5], "relevance": [0, 0, 0]})
    ans = [
        {
            "states": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "actions": np.array([3, 4, 5]),
            "rewards": np.array([0, 0, 0]),
            "rtgs": np.array([0, 0, 0]),
        }
    ]
    result = create_dataset(df, user_num=1, item_pad=0)
    for key, value in ans[0].items():
        assert (value == result[0][key]).all()


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
