import bisect
import random
from typing import List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    from torch.utils.data import Dataset


def set_seed(seed):
    """
    Set random seed in all dependicies
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StateActionReturnDataset(Dataset):
    """
    Create Dataset from user trajectories
    """

    def __init__(self, user_trajectory, trajectory_len):
        self.user_trajectory = user_trajectory
        self.trajectory_len = trajectory_len

        self.len = 0
        self.prefix_lens = [0]
        for trajectory in self.user_trajectory:
            self.len += max(1, len(trajectory["actions"]) - 30 + 1)
            self.prefix_lens.append(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        user_num = bisect.bisect_right(self.prefix_lens, idx) - 1
        start = idx - self.prefix_lens[user_num]

        user = self.user_trajectory[user_num]
        end = min(len(user["actions"]), start + self.trajectory_len)
        states = torch.tensor(np.array(user["states"][start:end]), dtype=torch.float32)
        actions = torch.tensor(user["actions"][start:end], dtype=torch.long)
        rtgs = torch.tensor(user["rtgs"][start:end], dtype=torch.float32)
        # strange logic but work
        timesteps = start

        return states, actions, rtgs, timesteps, user_num


class ValidateDataset(Dataset):
    """
    Dataset for Validation
    """

    def __init__(self, user_trajectory, max_context_len, val_users, val_items):
        self.user_trajectory = user_trajectory
        self.max_context_len = max_context_len
        self.val_users = val_users
        self.val_items = val_items

    def __len__(self):
        return len(self.val_users)

    def __getitem__(self, idx):
        user_idx = self.val_users[idx]
        user = self.user_trajectory[user_idx]
        if len(user["actions"]) <= self.max_context_len:
            start = 0
            end = -1
        else:
            end = -1
            start = end - self.max_context_len

        states = torch.tensor(
            np.array(user["states"][start - (start < 0) : end]),
            dtype=torch.float32,
        )
        actions = torch.tensor(user["actions"][start:end], dtype=torch.long)
        rtgs = torch.zeros((end - start + 1 if start < 0 else len(user["actions"])))
        rtgs[start:end] = torch.tensor(user["rtgs"][start:end], dtype=torch.float32)
        rtgs[end] = 10
        timesteps = len(user["actions"]) + start if start < 0 else 0

        return states, actions, rtgs, timesteps, user_idx


def pad_sequence(
    sequences: Union[torch.Tensor, List[torch.Tensor]],
    batch_first: bool = False,
    padding_value: float = 0.0,
    pos: str = "right",
) -> torch.Tensor:
    """
    Pad sequence
    """
    if pos == "right":
        padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)
    elif pos == "left":
        sequences = tuple(s.flip(0) for s in sequences)
        padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)
        _seq_dim = padded_sequence.dim()
        padded_sequence = padded_sequence.flip(-_seq_dim + batch_first)
    else:
        msg = f"pos should be either 'right' or 'left', but got {pos}"
        raise ValueError(msg)
    return padded_sequence


class Collator:
    """
    Callable class to merge several items to one batch
    """

    def __init__(self, item_pad):
        self.item_pad = item_pad

    def __call__(self, batch):
        states, actions, rtgs, timesteps, users_num = zip(*batch)

        return (
            pad_sequence(
                states,
                batch_first=True,
                padding_value=self.item_pad,
                pos="left",
            ),
            pad_sequence(
                actions,
                batch_first=True,
                padding_value=self.item_pad,
                pos="left",
            ).unsqueeze(-1),
            pad_sequence(rtgs, batch_first=True, padding_value=0, pos="left").unsqueeze(-1),
            torch.tensor(timesteps).unsqueeze(-1).unsqueeze(-1),
            torch.tensor(users_num).unsqueeze(-1),
        )


def matrix2df(matrix, users=None, items=None):
    """
    Creata DataFrame from matrix
    """
    users = np.arange(matrix.shape[0]) if users is None else np.array(users.cpu())
    if items is None:
        items = np.arange(matrix.shape[1])
    x1 = np.repeat(users, len(items))
    x2 = np.tile(items, len(users))
    x3 = np.array(matrix.cpu()).flatten()

    return pd.DataFrame(np.array([x1, x2, x3]).T, columns=["user_idx", "item_idx", "relevance"])


class WarmUpScheduler(_LRScheduler):
    """
    Implementation of WarmUp
    """

    def __init__(
        self,
        optimizer: Optimizer,
        dim_embed: int,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups


def calc_lr(step, dim_embed, warmup_steps):
    """
    Learning rate calculation
    """
    return dim_embed ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def create_dataset(
    df, user_num, item_pad, time_col="timestamp", user_col="user_idx", item_col="item_idx", relevance_col="relevance"
):
    """
    Create dataset from DataFrame
    """
    user_trajectory = [{} for _ in range(user_num)]
    df = df.sort_values(by=time_col)
    for user_idx in tqdm(range(user_num)):
        user_trajectory[user_idx]["states"] = [[item_pad, item_pad, item_pad]]
        user_trajectory[user_idx]["actions"] = []
        user_trajectory[user_idx]["rewards"] = []

        user = user_trajectory[user_idx]
        user_df = df[df[user_col] == user_idx]
        for _, row in user_df.iterrows():
            action = row[item_col]
            user["actions"].append(action)
            if row[relevance_col] > 3:
                user["rewards"].append(1)
                user["states"].append([user["states"][-1][1], user["states"][-1][2], action])
            else:
                user["rewards"].append(0)
                user["states"].append(user["states"][-1])

        user["rtgs"] = np.cumsum(user["rewards"][::-1])[::-1]
        for key in user:
            user[key] = np.array(user[key])

    return user_trajectory


# For debug
def fast_create_dataset(
    df,
    user_num,
    item_pad,
    time_field="timestamp",
    user_field="user_idx",
    item_field="item_idx",
    relevance_field="relevance",
):
    """
    Create dataset from DataFrame
    """
    user_trajectory = [{} for _ in range(user_num)]
    df = df.sort_values(by=time_field)
    for user_idx in tqdm(range(user_num)):
        user_trajectory[user_idx]["states"] = [[item_pad, item_pad, item_pad]]
        user_trajectory[user_idx]["actions"] = []
        user_trajectory[user_idx]["rewards"] = []

        user = user_trajectory[user_idx]
        user_df = df[df[user_field] == user_idx]
        for idx, (_, row) in enumerate(user_df.iterrows()):
            if idx >= 35:
                break
            action = row[item_field]
            user["actions"].append(action)
            if row[relevance_field] > 3:
                user["rewards"].append(1)
                user["states"].append([user["states"][-1][1], user["states"][-1][2], action])
            else:
                user["rewards"].append(0)
                user["states"].append(user["states"][-1])

        user["rtgs"] = np.cumsum(user["rewards"][::-1])[::-1]
        for key in user:
            user[key] = np.array(user[key])

    return user_trajectory
