import pickle
import tqdm
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pyspark.sql
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.utils.data as td
from pandas import DataFrame
from pytorch_ranger import Ranger
from tensorboardX import SummaryWriter

from replay.constants import REC_SCHEMA
from replay.models.base_torch_rec import TorchRecommender


def to_np(tensor):
    return tensor.detach().cpu().numpy()


class Buffer:
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, user, memory, action, reward, next_user, next_memory, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(
                (user, memory, action, reward, next_user, next_memory, done)
            )
        else:
            self.buffer[self.pos] = (
                user,
                memory,
                action,
                reward,
                next_user,
                next_memory,
                done,
            )

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        user = np.concatenate(batch[0])
        memory = np.concatenate(batch[1])
        action = batch[2]
        reward = batch[3]
        next_user = np.concatenate(batch[4])
        next_memory = np.concatenate(batch[5])
        done = batch[6]

        return user, memory, action, reward, next_user, next_memory, done

    def __len__(self):
        return len(self.buffer)


class Prioritized_Buffer:
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, user, memory, action, reward, next_user, next_memory, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(
                (user, memory, action, reward, next_user, next_memory, done)
            )
        else:
            self.buffer[self.pos] = (
                user,
                memory,
                action,
                reward,
                next_user,
                next_memory,
                done,
            )

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        user = np.concatenate(batch[0])
        memory = np.concatenate(batch[1])
        action = batch[2]
        reward = batch[3]
        next_user = np.concatenate(batch[4])
        next_memory = np.concatenate(batch[5])
        done = batch[6]

        return (
            user,
            memory,
            action,
            reward,
            next_user,
            next_memory,
            done,
            indices,
            weights,
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class EvalDataset(td.Dataset):
    def __init__(
        self,
        positive_data,
        item_num,
        positive_mat,
        negative_samples=48,
        seeds=None,
    ):
        super(EvalDataset, self).__init__()
        self.positive_data = np.array(positive_data)
        self.item_num = item_num
        self.positive_mat = positive_mat
        self.negative_samples = negative_samples
        self.seeds = seeds

        self.reset()

    def reset(self):
        print("Resetting dataset")
        data = self.create_valid_data()
        labels = np.zeros(
            len(self.positive_data) * (1 + self.negative_samples)
        )
        labels[:: 1 + self.negative_samples] = 1
        self.data = np.concatenate(
            [np.array(data), np.array(labels)[:, np.newaxis]], axis=1
        )

    def create_valid_data(self):
        valid_data = []

        if self.seeds is not None:
            for user, user_id, positive, *other in self.positive_data:
                if len(self.seeds[self.seeds["user_id"] == user_id]) >= 48:
                    valid_data.append([int(user), int(positive)])
                    for negative in (
                        self.seeds[self.seeds["user_id"] == user_id]
                        .sort_values("score")
                        .head(50)["item_idx"]
                    ):
                        valid_data.append([int(user), int(negative)])
        else:
            for user, user_id, positive, *other in self.positive_data:
                valid_data.append([int(user), int(positive)])
                for i in range(self.negative_samples):
                    negative = np.random.randint(self.item_num)
                    while (user, negative) in self.positive_mat:
                        negative = np.random.randint(self.item_num)

                    valid_data.append([int(user), int(negative)])
        return valid_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, item, label = self.data[idx]
        output = {
            "user": user,
            "item": item,
            "label": np.float32(label),
        }
        return output


class OUNoise:
    """https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py"""

    def __init__(
        self,
        action_dim,
        mu=0.0,
        theta=0.15,
        max_sigma=0.4,
        min_sigma=0.4,
        noise_type="ou",
        decay_period=10,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.noise_type = noise_type
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(
            self.action_dim
        )
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        if self.noise_type == "ou":
            ou_state = self.evolve_state()
            return torch.tensor([action + ou_state]).float()
        elif self.noise_type == "gauss":
            return torch.tensor(
                [self.sigma * np.random.randn(self.action_dim)]
            ).float()


class Actor_DRR(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.initialize()

    def initialize(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, state):
        return self.layers(state)

    def get_action(
        self,
        state_repr,
        action_emb,
        items,
        return_scores=False,
    ):
        scores = torch.bmm(
            state_repr.item_embeddings(items).unsqueeze(0),
            action_emb.T.unsqueeze(0),
        ).squeeze(0)
        if return_scores:
            return scores, torch.gather(items, 0, scores.argmax(0))
        else:
            return torch.gather(items, 0, scores.argmax(0))


class Critic_DRR(nn.Module):
    def __init__(self, state_repr_dim, action_emb_dim, hidden_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_repr_dim + action_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.initialize()

    def initialize(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.layers(x)
        return x


# pylint: disable=too-many-instance-attributes
class Env:
    def __init__(self, user_item_matrix, item_num, user_num, N):
        """
        Memory is initialized as ['item_num'] * 'N' for each user.
        It is padding indexes in state_repr and will result in zero embeddings.
        """
        self.matrix = user_item_matrix
        self.item_count = item_num
        self.user_count = user_num
        self.N = N
        self.memory = np.ones([user_num, N]) * item_num

    def reset(self, user_id):
        self.user_id = user_id
        self.viewed_items = []
        self.related_items = list(
            np.argwhere(self.matrix[self.user_id] > 0)[:, 1]
        )
        self.num_rele = len(self.related_items)
        self.nonrelated_items = list(
            np.argwhere(self.matrix[self.user_id] < 0)[:, 1]
        )
        self.available_items = self.related_items + self.nonrelated_items

        return torch.tensor([self.user_id]), torch.tensor(
            self.memory[[self.user_id], :]
        )

    def reset_old(self, user_id):
        self.user_id = user_id
        self.viewed_items = []
        self.related_items = np.argwhere(self.matrix[self.user_id] > 0)[:, 1]
        self.num_rele = len(self.related_items)
        self.nonrelated_items = np.random.choice(
            list(set(range(self.item_count)) - set(self.related_items)),
            self.num_rele,
        )
        self.available_items = np.zeros(self.num_rele * 2)
        self.available_items[::2] = self.related_items
        self.available_items[1::2] = self.nonrelated_items

        return torch.tensor([self.user_id]), torch.tensor(
            self.memory[[self.user_id], :]
        )

    def reset_memory(self):
        self.memory = np.ones([self.item_count, self.N]) * self.user_count

    def step(self, action, action_emb=None, buffer=None):
        initial_user = self.user_id
        initial_memory = self.memory[[initial_user], :]

        reward = float(to_np(action)[0] in self.related_items)
        if reward:
            if len(action) == 1:
                self.memory[self.user_id] = list(
                    self.memory[self.user_id][1:]
                ) + [action]
            else:
                self.memory[self.user_id] = list(
                    self.memory[self.user_id][1:]
                ) + [action[0]]

        self.viewed_items.append(to_np(action)[0])
        if len(self.viewed_items) == len(self.related_items):
            done = 1
        else:
            done = 0

        if buffer is not None:
            buffer.push(
                np.array([initial_user]),
                np.array(initial_memory),
                to_np(action_emb)[0],
                np.array([reward]),
                np.array([self.user_id]),
                self.memory[[self.user_id], :],
                np.array([reward]),
            )

        return (
            torch.tensor([self.user_id]),
            torch.tensor(self.memory[[self.user_id], :]),
            reward,
            done,
        )


class State_Repr_Module(nn.Module):
    def __init__(
        self, user_num, item_num, embedding_dim, N, user_embeddings=None, item_embeddings=None
    ):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(
            item_num + 1, embedding_dim, padding_idx=int(item_num)
        )
        self.drr_ave = torch.nn.Conv1d(
            in_channels=N, out_channels=1, kernel_size=1
        )

        self.initialize(user_embeddings, item_embeddings)

    def initialize(self, user_embeddings, item_embeddings):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data[-1].zero_()

        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.uniform_(self.drr_ave.weight)

        self.drr_ave.bias.data.zero_()

    def forward(self, user, memory):
        user_embedding = self.user_embeddings(user.long())

        item_embeddings = self.item_embeddings(memory.long())
        drr_ave = self.drr_ave(item_embeddings).squeeze(1)

        return torch.cat(
            (user_embedding, user_embedding * drr_ave, drr_ave), 1
        )


# pylint: disable=too-many-arguments
class DDPG(TorchRecommender):
    _search_space = {
        "gamma": {"type": "uniform", "args": [0.8, 0.8]},
    }

    def __init__(
        self,
        batch_size=512,
        embedding_dim=8,
        hidden_dim=16,
        N=5,
        noise_sigma=0.4,
        PER=True,
        value_lr=1e-5,
        value_decay=1e-5,
        policy_lr=1e-5,
        policy_decay=1e-6,
        state_repr_lr=1e-5,
        state_repr_decay=1e-3,
        log_dir="data/logs/duration/tmp",
        gamma=0.8,
        min_value=-10,
        max_value=10,
        soft_tau=1e-3,
        buffer_size=1000000000,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.N = N
        self.PER = PER
        self.value_lr = value_lr
        self.value_decay = value_decay
        self.policy_lr = policy_lr
        self.policy_decay = policy_decay
        self.state_repr_lr = state_repr_lr
        self.state_repr_decay = state_repr_decay
        self.log_dir = Path(log_dir)
        self.gamma = gamma
        self.min_value = min_value
        self.max_value = max_value
        self.soft_tau = soft_tau
        self.buffer_size = buffer_size
        self.ou_noise = OUNoise(
            self.embedding_dim,
            max_sigma=noise_sigma,
            min_sigma=noise_sigma,
            # noise_type="gauss",
        )  # , decay_period=1000000)

    @property
    def _init_args(self):
        return {
            "gamma": self.gamma,
        }

    def _batch_pass(self, batch, model) -> Dict[str, Any]:
        pass

    def _loss(self, **kwargs) -> torch.Tensor:
        pass

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: pyspark.sql.DataFrame,
        k: int,
        users: pyspark.sql.DataFrame,
        items: pyspark.sql.DataFrame,
        user_features: Optional[pyspark.sql.DataFrame] = None,
        item_features: Optional[pyspark.sql.DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> pyspark.sql.DataFrame:
        items_consider_in_pred = items.toPandas()["item_idx"].values
        items_count = self._item_dim
        policy_net = self.policy_net.cpu()
        state_repr = self.state_repr.cpu()
        memory = self.train_env.memory
        agg_fn = self._predict_by_user_ddpg

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            return agg_fn(
                pandas_df,
                policy_net,
                state_repr,
                memory,
                items_consider_in_pred,
                k,
                items_count,
            )[["user_idx", "item_idx", "relevance"]]

        self.logger.debug("Predict started")
        # do not apply map on cold users for MultVAE predict
        join_type = "inner" if self.__str__() == "MultVAE" else "left"
        recs = (
            users.join(log, how=join_type, on="user_idx")
            .select("user_idx", "item_idx")
            .groupby("user_idx")
            .applyInPandas(grouped_map, REC_SCHEMA)
        )
        return recs

    @staticmethod
    def _predict_pairs_inner(
        policy_net,
        state_repr,
        memory,
        user_idx: int,
        items_np: np.ndarray,  # = torch.tensor([i for i in range(item_num)],
        cnt: Optional[int] = None,
    ) -> DataFrame:
        with torch.no_grad():
            user_batch = torch.LongTensor([user_idx])
            action_emb = policy_net(
                state_repr(
                    user_batch,
                    torch.tensor(memory)[to_np(user_batch).astype(int), :],
                )
            )
            user_recs, _ = policy_net.get_action(
                state_repr,
                action_emb,
                torch.tensor(items_np),
                return_scores=True,
            )
            user_recs = user_recs.squeeze(1)

            if cnt is not None:
                # user_recs, ind = user_recs.topk(cnt, dim=0)
                # items_np = torch.take(items_np, ind.T).cpu().numpy().tolist()
                best_item_idx = (
                    torch.argsort(user_recs, descending=True)[:cnt]
                ).numpy()
                user_recs = user_recs[best_item_idx]
                items_np = items_np[best_item_idx]

            return pd.DataFrame(
                {
                    "user_idx": user_recs.shape[0] * [user_idx],
                    "item_idx": items_np,
                    "relevance": user_recs,
                }
            )

    # pylint: disable=arguments-differ
    def _predict_by_user(
        pandas_df: pd.DataFrame,
        model: nn.Module,
        items_np: np.ndarray,
        k: int,
        item_count: int,
    ) -> pd.DataFrame:
        pass

    @staticmethod
    def _predict_by_user_ddpg(
        pandas_df: pd.DataFrame,
        policy_net: nn.Module,
        state_repr: nn.Module,
        memory,
        items_np: np.ndarray,
        k: int,
        item_count: int,
    ):
        return DDPG._predict_pairs_inner(
            policy_net,
            state_repr,
            memory,
            user_idx=pandas_df["user_idx"][0],
            items_np=items_np,
            cnt=min(len(pandas_df) + k, len(items_np)),
        )

    @staticmethod
    def _predict_by_user_pairs(
        pandas_df: pd.DataFrame,
        model: nn.Module,
        item_count: int,
    ):
        pass

    def _get_data_loader(self, data, item_num, matrix, seeds=None):
        if seeds is not None:
            dataset = EvalDataset(data, item_num, matrix, seeds=seeds)
        else:
            dataset = EvalDataset(data, item_num, matrix)

        loader = td.DataLoader(dataset, batch_size=100, shuffle=False)
        return loader

    def get_beta(self, idx, beta_start=0.4, beta_steps=100000):
        return min(1.0, beta_start + idx * (1.0 - beta_start) / beta_steps)

    def preprocess_log(self, log):
        data = log.toPandas()

        data = data[data["relevance"] > 0][["user_idx", "item_idx"]]
        user_num = data["user_idx"].max() + 1
        item_num = data["item_idx"].max() + 1

        train_data = data.sample(frac=0.9, random_state=16)
        test_data = data.drop(train_data.index).values.tolist()
        train_data = train_data.values.tolist()

        train_mat = defaultdict(int)
        test_mat = defaultdict(int)
        for user, item in train_data:
            train_mat[user, item] = 1.0
        for user, item in test_data:
            test_mat[user, item] = 1.0
        train_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        dict.update(train_matrix, train_mat)
        test_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        dict.update(test_matrix, test_mat)

        appropriate_users = np.arange(user_num).reshape(-1, 1)[
            (train_matrix.sum(1) >= 10)
        ]

        return (
            train_matrix,
            test_data,
            test_matrix,
            user_num,
            item_num,
            appropriate_users,
        )

    def preprocess_sound(self, log):
        data = log.toPandas()

        data = data.drop("proc_dt", axis=1)
        #         data = data[data["relevance"] > 0].drop('proc_dt', axis=1)
        user_num = data["user_idx"].max() + 1
        item_num = data["item_idx"].max() + 1

        train_data = data.sample(
            frac=0.9, random_state=16
        )  # TODO [data["relevance"] > 0]
        appropriate_users = (
            train_data["user_idx"]
            .value_counts()[train_data["user_idx"].value_counts() > 10]
            .index
        )
        test_data = data.drop(train_data.index).values.tolist()
        train_data = train_data.values.tolist()

        train_mat = defaultdict(float)
        test_mat = defaultdict(float)
        for user, _, item, _, rel, *emb in train_data:
            train_mat[user, item] = rel
        for user, _, item, _, rel, *emb in test_data:
            test_mat[user, item] = rel
        train_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        dict.update(train_matrix, train_mat)
        test_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        dict.update(test_matrix, test_mat)

        #         appropriate_users = np.arange(user_num)
        #         appropriate_users = np.arange(user_num).reshape(-1, 1)[
        #             (train_matrix.sum(1) >= 10)
        #         ]
        user_embeddings = (
            data.set_index("user_idx").iloc[:, -8:].drop_duplicates()
        )

        return (
            train_matrix,
            test_data,
            test_matrix,
            user_num,
            item_num,
            appropriate_users,
            user_embeddings,
        )

    def preprocess_sound_val(self, log):
        data = log.toPandas()

        data = data[data["relevance"] > 0].drop("proc_dt", axis=1)
        #         data["user_idx"] = data["user_idx"].astype(int)
        #         data["item_idx"] = data["item_idx"].astype(int)
        user_num = data["user_idx"].max() + 1
        item_num = data["item_idx"].max() + 1

        train_data = data.sample(
            frac=0.9, random_state=16
        )  # TODO [data["relevance"] > 0]
        appropriate_users = (
            train_data["user_idx"]
            .value_counts()[train_data["user_idx"].value_counts() > 10]
            .index
        )
        test_df = data.drop(train_data.index)
        test_data = data.drop(train_data.index).values.tolist()
        train_data = train_data.values.tolist()

        train_mat = defaultdict(float)
        test_mat = defaultdict(float)
        for user, _, item, _, rel, *_ in train_data:
            train_mat[user, item] = rel
        for user, _, item, _, rel, *_ in test_data:
            test_mat[user, item] = rel
        train_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        dict.update(train_matrix, train_mat)
        test_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        dict.update(test_matrix, test_mat)

        user_embeddings = (
            data.set_index("user_idx").iloc[:, -8:].drop_duplicates()
        )

        return (
            train_matrix,
            test_data,
            test_df,
            test_matrix,
            user_num,
            item_num,
            appropriate_users,
            user_embeddings,
        )

    def hit_metric(self, recommended, actual):
        return int(actual in recommended)

    def dcg_metric(self, recommended, actual):
        if actual in recommended:
            index = recommended.index(actual)
            return np.reciprocal(np.log2(index + 2))
        return 0

    def _ddpg_update(
        self,
        policy_optimizer,
        state_repr_optimizer,
        value_optimizer,
        replay_buffer,
        writer=None,
        step=0,
    ):
        beta = self.get_beta(step)
        (
            user,
            memory,
            action,
            reward,
            next_user,
            next_memory,
            done,
        ) = replay_buffer.sample(self.batch_size, beta)
        # user, memory, action, reward, next_user, next_memory, done, indices, weights = replay_buffer.sample(self.batch_size, beta)
        user = torch.FloatTensor(user)
        memory = torch.FloatTensor(memory)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_user = torch.FloatTensor(next_user)
        next_memory = torch.FloatTensor(next_memory)
        done = torch.FloatTensor(done)
        # weights     = torch.FloatTensor(weights)

        state = self.state_repr(user, memory)
        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_state = self.state_repr(next_user, next_memory)
        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(
            expected_value, self.min_value, self.max_value
        )

        value = self.value_net(state, action)
        value_loss = (value - expected_value.detach()).squeeze(1).pow(2).mean()
        # value_loss = (value - expected_value.detach()).squeeze(1).pow(2) * weights # .mean()
        # prios = value_loss + 1e-5
        # value_loss  = value_loss.mean()

        state_repr_optimizer.zero_grad()
        policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        policy_optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        # replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        value_optimizer.step()
        state_repr_optimizer.step()

        for target_param, param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau)
                + param.data * self.soft_tau
            )
        for target_param, param in zip(
            self.target_policy_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau)
                + param.data * self.soft_tau
            )

        if writer:
            writer.add_histogram("value", value, step)
            writer.add_histogram("target_value", target_value, step)
            writer.add_histogram("expected_value", expected_value, step)
            writer.add_histogram("policy_loss", -policy_loss, step)
            writer.add_histogram("value_loss", value_loss, step)

    def run_evaluation(
        self,
        net,
        state_representation,
        training_env_memory,
        test_env,
        loader,
    ):
        hits3 = []
        hits = []
        dcgs3 = []
        dcgs = []
        test_env.memory = training_env_memory.copy()
        user, memory = test_env.reset(
            int(to_np(next(iter(loader))["user"])[0])
        )
        for batch in loader:
            action_emb = net(state_representation(user, memory))
            scores, action = net.get_action(
                state_representation,
                action_emb,
                batch["item"].long(),
                return_scores=True,
            )
            user, memory, _, _ = test_env.step(action)

            _, ind = scores[:, 0].topk(3)
            predictions = batch["item"].take(ind).cpu().numpy().tolist()
            actual = batch["item"][0].item()
            hits3.append(self.hit_metric(predictions, actual))
            dcgs3.append(self.dcg_metric(predictions, actual))

            _, ind = scores[:, 0].topk(1)
            predictions = batch["item"].take(ind).cpu().numpy().tolist()
            hits.append(self.hit_metric(predictions, actual))
            dcgs.append(self.dcg_metric(predictions, actual))
        return np.mean(hits), np.mean(dcgs), np.mean(hits3), np.mean(dcgs3)

    def load_user_embeddings(self, user_embeddings_path):
        user_embeddings = pd.read_parquet(user_embeddings_path)
        indexes = user_embeddings["user_idx"]
        embeddings = user_embeddings.drop("user_idx", axis=1)
        self.state_repr.user_embeddings.weight.data[indexes].copy_(
            torch.from_numpy(embeddings.values)
        )

    def load_item_embeddings(self, item_embeddings_path):
        item_embeddings = pd.read_parquet(item_embeddings_path)
        indexes = item_embeddings["item_idx"]
        embeddings = item_embeddings.drop("item_idx", axis=1)
        self.state_repr.item_embeddings.weight.data[indexes].copy_(
            torch.from_numpy(embeddings.values)
        )

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        (
            train_matrix,
            test_data,
            test_matrix,
            user_num,
            item_num,
            appropriate_users,
            user_embeddings,
        ) = self.preprocess_sound(log)
        np.random.seed(16)
        torch.manual_seed(16)
        users = np.random.permutation(appropriate_users)
        self.train_env = Env(train_matrix, item_num, user_num, N=self.N)
        self.test_env = Env(test_matrix, item_num, user_num, N=self.N)
        valid_loader = None  # self._get_data_loader(
        #             np.array(test_data)[np.array(test_data)[:, 0] == '16'],
        #             item_num,
        #             test_matrix,
        #         )

        self.state_repr = State_Repr_Module(
            user_num, item_num, self.embedding_dim, self.N, user_embeddings
        )
        self.policy_net = Actor_DRR(self.embedding_dim, self.hidden_dim)
        self.value_net = Critic_DRR(
            self.embedding_dim * 3, self.embedding_dim, self.hidden_dim
        )
        self.target_value_net = Critic_DRR(
            self.embedding_dim * 3, self.embedding_dim, self.hidden_dim
        )
        self.target_policy_net = Actor_DRR(self.embedding_dim, self.hidden_dim)
        for target_param, param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(param.data)
        for target_param, param in zip(
            self.target_policy_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(param.data)
        policy_optimizer = Ranger(
            self.policy_net.parameters(),
            lr=self.policy_lr,
            weight_decay=self.policy_decay,
        )
        state_repr_optimizer = Ranger(
            self.state_repr.parameters(),
            lr=self.state_repr_lr,
            weight_decay=self.state_repr_decay,
        )
        value_optimizer = Ranger(
            self.value_net.parameters(),
            lr=self.value_lr,
            weight_decay=self.value_decay,
        )

        replay_buffer = Buffer(self.buffer_size)
        writer = SummaryWriter(log_dir=self.log_dir)

        self.train(
            users,
            policy_optimizer,
            state_repr_optimizer,
            value_optimizer,
            replay_buffer,
            writer,
            valid_loader,
        )

    def train(
        self,
        users,
        policy_optimizer,
        state_repr_optimizer,
        value_optimizer,
        replay_buffer,
        writer,
        valid_loader=None,
    ):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        hits, dcgs, rewards, steps = [], [], [], []
        step, best_step = 0, 0

        for i, u in enumerate(tqdm.tqdm(users)):
            user, memory = self.train_env.reset(u)
            self.ou_noise.reset()

            for t in range(len(self.train_env.related_items)):
                action_emb = self.policy_net(self.state_repr(user, memory))
                action_emb = self.ou_noise.get_action(to_np(action_emb)[0], t)
                action = self.policy_net.get_action(
                    self.state_repr,
                    action_emb,
                    torch.tensor(
                        [
                            item
                            for item in self.train_env.available_items
                            if item not in self.train_env.viewed_items
                        ]
                    ).long(),
                )
                user, memory, reward, _ = self.train_env.step(
                    action, action_emb, replay_buffer
                )
                rewards.append(reward)
                if len(steps) < i and reward > 0:
                    steps.append(t)

                if len(replay_buffer) > self.batch_size:
                    self._ddpg_update(
                        policy_optimizer,
                        state_repr_optimizer,
                        value_optimizer,
                        replay_buffer,
                        writer=writer,
                        step=step,
                    )

                if step % 10000 == 0 and step > 0:
                    torch.save(
                        self.policy_net.state_dict(),
                        self.log_dir / f"policy_net_{step}.pth",
                    )
                    torch.save(
                        self.state_repr.state_dict(),
                        self.log_dir / f"state_repr_{step}.pth",
                    )
                    if valid_loader:
                        self.test_env.reset_memory()
                        hit, dcg, hit3, dcg3 = self.run_evaluation(
                            self.policy_net,
                            self.state_repr,
                            self.train_env.memory,
                            self.test_env,
                            loader=valid_loader
                        )
                        if writer:
                            writer.add_scalar('hit', hit, step)
                            writer.add_scalar('dcg', dcg, step)
                        hits.append(hit)
                        dcgs.append(dcg)
                        if np.mean(np.array([hit, dcg]) - np.array([hits[best_step], dcgs[best_step]])) > 0:
                            best_step = step // 10000
                            torch.save(self.policy_net.state_dict(), self.log_dir / 'best_policy_net.pth')
                            torch.save(self.state_repr.state_dict(), self.log_dir / 'best_state_repr.pth')
                step += 1
        #             if len(steps) < i:
        #                 steps.append(t)
        #             with torch.no_grad():
        #                 if steps and writer:
        #                     writer.add_scalars('convergence_steps', {'max': np.max(steps)}, step)
        #                     writer.add_scalars('convergence_steps', {'current': steps[-1]}, step)
        #                     writer.add_histogram('reward_per_episode', np.mean(rewards[-100:]), step)

        torch.save(
            self.policy_net.state_dict(),
            self.log_dir / "policy_net_final.pth",
        )
        torch.save(
            self.state_repr.state_dict(),
            self.log_dir / "state_repr_final.pth",
        )
        with open(self.log_dir / "memory.pickle", "wb") as f:
            pickle.dump(self.train_env.memory, f)
        writer.close()
