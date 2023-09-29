import bisect
import random
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def drop_cold_users(df):
    actions_cnt = df.groupby(by="user_id", as_index=False).count()
    cold_users = actions_cnt[actions_cnt["item_id"] < 35]["user_id"]
    new_df = df[df["user_id"].isin(cold_users) == False]

    return new_df


def reindex(df):
    item_encoder = LabelEncoder().fit(df["item_idx"])
    df["item_idx"] = item_encoder.transform(df["item_idx"])

    user_encoder = LabelEncoder().fit(df["user_idx"])
    df["user_idx"] = user_encoder.transform(df["user_idx"])

    return df


def leave_last_out(data, userid="user_id", timeid="timestamp"):
    sorted = data.sort_values(timeid)
    holdout = sorted.drop_duplicates(subset=[userid], keep="last")
    remaining = data.drop(holdout.index)
    return remaining, holdout


def model_evaluate(recommended_items, holdout, holdout_description, topn=10):
    itemid = holdout_description["items"]
    holdout_items = holdout[itemid].values
    assert recommended_items.shape[0] == len(holdout_items)
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
    # HR calculation
    hr = np.mean(hits_mask.any(axis=1))
    # MRR calculation
    n_test_users = recommended_items.shape[0]
    _, hit_rank = np.where(hits_mask)
    mrr = np.sum(1.0 / (hit_rank + 1)) / n_test_users
    return hr, mrr


@torch.no_grad()
def sample(
    model,
    x,
    steps,
    temperature=1.0,
    sample=False,
    top_k=None,
    actions=None,
    rtgs=None,
    timesteps=None,
):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x if x.size(1) <= block_size // 3 else x[:, -block_size // 3 :]  # crop context if needed
        if actions is not None:
            actions = (
                actions if actions.size(1) <= block_size // 3 else actions[:, -block_size // 3 :]
            )  # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size // 3 else rtgs[:, -block_size // 3 :]  # crop context if needed
        logits, _ = model(
            x_cond,
            actions=actions,
            targets=None,
            rtgs=rtgs,
            timesteps=timesteps,
        )
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        x = ix

    return x


# For debug
class FastStateActionReturnDataset(Dataset):
    def __init__(self, user_trajectory, trajectory_len):
        self.user_trajectory = user_trajectory
        self.trajectory_len = trajectory_len

    def __len__(self):
        return len(self.user_trajectory)

    def __getitem__(self, idx):
        user = self.user_trajectory[idx]
        start = 0
        end = min(len(user["actions"]), start + self.trajectory_len)
        states = torch.tensor(np.array(user["states"][start:end]), dtype=torch.float32)
        actions = torch.tensor(user["actions"][start:end], dtype=torch.long)
        rtgs = torch.tensor(user["rtgs"][start:end], dtype=torch.float32)
        # strange logic but work
        timesteps = start

        return states, actions, rtgs, timesteps, idx


class StateActionReturnDataset(Dataset):
    def __init__(self, user_trajectory, trajectory_len):
        self.user_trajectory = user_trajectory
        self.trajectory_len = trajectory_len

        self.len = 0
        self.prefix_lens = [0]
        for trajectory in self.user_trajectory:
            # print(f'{trajectory=}')
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
    if pos == "right":
        padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)
    elif pos == "left":
        sequences = tuple(map(lambda s: s.flip(0), sequences))
        padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)
        _seq_dim = padded_sequence.dim()
        padded_sequence = padded_sequence.flip(-_seq_dim + batch_first)
    else:
        raise ValueError("pos should be either 'right' or 'left', but got {}".format(pos))
    return padded_sequence


class Collator:
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
    HEADER = ["user_idx", "item_idx", "relevance"]
    if users is None:
        users = np.arange(matrix.shape[0])
    else:
        users = np.array(users.cpu())
    if items is None:
        items = np.arange(matrix.shape[1])
    x1 = np.repeat(users, len(items))
    x2 = np.tile(items, len(users))
    x3 = np.array(matrix.cpu()).flatten()

    return pd.DataFrame(np.array([x1, x2, x3]).T, columns=HEADER)


class WarmUpScheduler(_LRScheduler):
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
    return dim_embed ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def create_dataset(
    df, user_num, item_pad, time_col="timestamp", user_col="user_idx", item_col="item_idx", relevance_col="relevance"
):
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
