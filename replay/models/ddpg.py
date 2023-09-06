import tqdm
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pandas import DataFrame
from pyspark.sql import functions as sf
from pytorch_ranger import Ranger
from torch import nn

from replay.data import REC_SCHEMA
from replay.models.base_torch_rec import Recommender


def to_np(tensor: torch.Tensor) -> np.array:
    """Converts torch.Tensor to numpy."""
    return tensor.detach().cpu().numpy()


class ReplayBuffer:
    """
    Stores transitions for training RL model.

    Usually transition is (state, action, reward, next_state).
    In this implementation we compute state using embedding of user
    and embeddings of `memory_size` latest relevant items.
    Thereby in this ReplayBuffer we store (user, memory) instead of state.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, capacity: int = 1000000, prob_alpha: float = 0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = {
            "user": [],
            "memory": [],
            "action": [],
            "reward": [],
            "next_user": [],
            "next_memory": [],
            "done": [],
        }
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, user, memory, action, reward, next_user, next_memory, done):
        """Add transition to buffer."""
        max_priority = (
            self.priorities.max() if len(self.buffer["user"]) > 0 else 1.0
        )

        if len(self.buffer) < self.capacity:
            self.buffer["user"].append(user)
            self.buffer["memory"].append(memory)
            self.buffer["action"].append(action)
            self.buffer["reward"].append(reward)
            self.buffer["next_user"].append(next_user)
            self.buffer["next_memory"].append(next_memory)
            self.buffer["done"].append(done)
        else:
            self.buffer["user"][self.pos] = user
            self.buffer["memory"][self.pos] = memory
            self.buffer["action"][self.pos] = action
            self.buffer["reward"][self.pos] = reward
            self.buffer["next_user"][self.pos] = next_user
            self.buffer["next_memory"][self.pos] = next_memory
            self.buffer["done"][self.pos] = done

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    # pylint: disable=too-many-locals
    def sample(self, batch_size, beta=0.4):
        """Sample transition from buffer."""
        current_buffer_len = len(self.buffer["user"])
        if current_buffer_len == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.pos]

        probs = priorities ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(current_buffer_len, batch_size, p=probs)

        weights = (current_buffer_len * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return {
            "user": np.concatenate([self.buffer["user"][i] for i in indices]),
            "memory": np.concatenate(
                [self.buffer["memory"][i] for i in indices]
            ),
            "action": np.array([self.buffer["action"][i] for i in indices]),
            "reward": np.array([self.buffer["reward"][i] for i in indices]),
            "next_user": np.concatenate(
                [self.buffer["next_user"][i] for i in indices]
            ),
            "next_memory": np.concatenate(
                [self.buffer["next_memory"][i] for i in indices]
            ),
            "done": np.array([self.buffer["done"][i] for i in indices]),
        }

    def __len__(self):
        return len(self.buffer["user"])


# pylint: disable=too-many-instance-attributes,too-many-arguments,not-callable
class OUNoise:
    """https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py"""

    def __init__(
        self,
        action_dim,
        theta=0.15,
        max_sigma=0.4,
        min_sigma=0.4,
        noise_type="ou",
        decay_period=10,
    ):
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.noise_type = noise_type
        self.state = np.zeros(action_dim)

    def reset(self):
        """Fill state with zeros."""
        self.state = np.zeros(self.action_dim)

    def evolve_state(self):
        """Perform OU discrete approximation step"""
        x = self.state
        d_x = -self.theta * x + self.sigma * np.random.randn(self.action_dim)
        self.state = x + d_x
        return self.state

    def get_action(self, action, step=0):
        """Get state after applying noise."""
        action = to_np(action)
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, step / self.decay_period
        )
        if self.noise_type == "ou":
            ou_state = self.evolve_state()
            return torch.tensor([action + ou_state]).float()
        elif self.noise_type == "gauss":
            return torch.tensor(
                [self.sigma * np.random.randn(self.action_dim)]
            ).float()
        else:
            raise ValueError("noise_type must be one of ['ou', 'gauss']")


class ActorDRR(nn.Module):
    """
    DDPG Actor model (based on `DRR
    <https://arxiv.org/pdf/1802.05814.pdf>`_).
    """

    def __init__(
        self, user_num, item_num, embedding_dim, hidden_dim, memory_size
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.state_repr = StateReprModule(
            user_num, item_num, embedding_dim, memory_size
        )

        self.initialize()

        self.environment = Env(item_num, user_num, memory_size)

    def initialize(self):
        """weight init"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, user, memory):
        """
        :param user: user batch
        :param memory: memory batch
        :return: output, vector of the size `embedding_dim`
        """
        state = self.state_repr(user, memory)
        return self.layers(state)

    # pylint: disable=not-callable
    def get_action(self, action_emb, items, return_scores=False):
        """
        :param action_emb: output of the .forward()
        :param items: items batch
        :param return_scores: whether to return scores of items
        :return: output, prediction (and scores if return_scores)
        """
        items = torch.tensor(items).long()
        scores = torch.bmm(
            self.state_repr.item_embeddings(items).unsqueeze(0),
            action_emb.T.unsqueeze(0),
        ).squeeze(0)
        if return_scores:
            return scores, torch.gather(items, 0, scores.argmax(0))
        else:
            return torch.gather(items, 0, scores.argmax(0))


class CriticDRR(nn.Module):
    """
    DDPG Critic model (based on `DRR
    <https://arxiv.org/pdf/1802.05814.pdf>`_).
    """

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
        """weight init"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, state, action):
        """
        :param state: state batch
        :param action: action batch
        :return: x, Q values for given states and actions
        """
        x = torch.cat([state, action], 1)
        x = self.layers(x)
        return x


# pylint: disable=too-many-instance-attributes, not-callable
class Env:
    """
    RL environment for recommender systems.

    Keep users' latest relevant items (memory).
    """

    matrix: np.array
    related_items: np.array
    nonrelated_items: np.array
    available_items: list
    user_id: int
    num_rele: int

    def __init__(self, item_num, user_num, memory_size):
        """
        Initialize memory as ['item_num'] * 'memory_size' for each user.

        'item_num' is a padding index in StateReprModule.
        It will result in zero embeddings.

        :param item_num: number of items
        :param user_num: number of users
        :param memory_size: maximum number of items in memory
        :param memory: np.array with users' latest relevant items
        :param matrix: sparse matrix with users-item ratings
        :param user_id: users_id number
        :param related_items: relevant items for user_id
        :param nonrelated_items: non-relevant items for user_id
        :param num_rele: number of related_items
        :param available_items: non-seen items
        """
        self.item_count = item_num
        self.user_count = user_num
        self.memory_size = memory_size
        self.memory = np.ones([user_num, memory_size]) * item_num

    def update_env(self, matrix=None, item_count=None):
        """Update some of Env attributes."""
        if item_count is not None:
            self.item_count = item_count
        if matrix is not None:
            self.matrix = matrix.copy()

    def reset(self, user_id):
        """
        :param user_id: user_id number
        :return: user, memory
        """
        self.user_id = user_id
        self.related_items = np.argwhere(self.matrix[self.user_id] > 0)[:, 1]
        self.num_rele = len(self.related_items)
        self.nonrelated_items = np.random.choice(
            list(set(range(self.item_count)) - set(self.related_items)),
            self.num_rele,
        )
        self.available_items = list(np.zeros(self.num_rele * 2))
        self.available_items[::2] = self.related_items
        self.available_items[1::2] = self.nonrelated_items

        return torch.tensor([self.user_id]), torch.tensor(
            self.memory[[self.user_id], :]
        )

    def step(self, action, action_emb=None, buffer=None):
        """Execute step and return (user, memory) for new state"""
        initial_user = self.user_id
        initial_memory = self.memory[[initial_user], :]

        reward = float(to_np(action[0]) in self.related_items)
        if reward:
            self.memory[self.user_id] = list(self.memory[self.user_id][1:]) + [
                action
            ]

        self.available_items.remove(to_np(action[0]))

        if buffer is not None:
            buffer.push(
                np.array([initial_user]),
                np.array(initial_memory),
                to_np(action_emb[0]),
                np.array([reward]),
                np.array([self.user_id]),
                self.memory[[self.user_id], :],
                np.array([reward]),
            )

        return (
            torch.tensor([self.user_id]),
            torch.tensor(self.memory[[self.user_id], :]),
            reward,
            0,
        )


class StateReprModule(nn.Module):
    """
    Compute state for RL environment. Based on `DRR paper
    <https://arxiv.org/pdf/1810.12027.pdf>`_

    Computes State is a concatenation of user embedding,
    weighted average pooling of `memory_size` latest relevant items
    and their pairwise product.
    """

    def __init__(
        self,
        user_num,
        item_num,
        embedding_dim,
        memory_size,
    ):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(
            item_num + 1, embedding_dim, padding_idx=int(item_num)
        )
        self.drr_ave = torch.nn.Conv1d(
            in_channels=memory_size, out_channels=1, kernel_size=1
        )

        self.initialize()

    def initialize(self):
        """weight init"""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data[-1].zero_()

        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.uniform_(self.drr_ave.weight)

        self.drr_ave.bias.data.zero_()

    def forward(self, user, memory):
        """
        :param user: user batch
        :param memory: memory batch
        :return: vector of dimension 3 * embedding_dim
        """
        user_embedding = self.user_embeddings(user.long())

        item_embeddings = self.item_embeddings(memory.long())
        drr_ave = self.drr_ave(item_embeddings).squeeze(1)

        return torch.cat(
            (user_embedding, user_embedding * drr_ave, drr_ave), 1
        )


# pylint: disable=too-many-arguments
class DDPG(Recommender):
    """
    `Deep Deterministic Policy Gradient
    <https://arxiv.org/pdf/1810.12027.pdf>`_

    This implementation enhanced by more advanced noise strategy.
    """

    batch_size: int = 512
    embedding_dim: int = 8
    hidden_dim: int = 16
    value_lr: float = 1e-5
    value_decay: float = 1e-5
    policy_lr: float = 1e-5
    policy_decay: float = 1e-6
    gamma: float = 0.8
    memory_size: int = 5
    min_value: int = -10
    max_value: int = 10
    buffer_size: int = 1000000
    _search_space = {
        "noise_sigma": {"type": "uniform", "args": [0.1, 0.6]},
        "noise_theta": {"type": "uniform", "args": [0.05, 0.15]},
    }
    checkpoint_step: int = 10000
    replay_buffer: ReplayBuffer
    ou_noise: OUNoise
    model: ActorDRR
    target_model: ActorDRR
    value_net: CriticDRR
    target_value_net: CriticDRR
    policy_optimizer: Ranger
    value_optimizer: Ranger

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        noise_sigma: float = 0.2,
        noise_theta: float = 0.05,
        noise_type: str = "ou",
        seed: int = 9,
        user_num: int = 10,
        item_num: int = 10,
        log_dir: str = "logs/tmp",
        exact_embeddings_size=True,
    ):
        """
        :param noise_sigma: Ornstein-Uhlenbeck noise sigma value
        :param noise_theta: Ornstein-Uhlenbeck noise theta value
        :param noise_type: type of action noise, one of ["ou", "gauss"]
        :param seed: random seed
        :param user_num: number of users, specify when using ``exact_embeddings_size``
        :param item_num: number of items, specify when using ``exact_embeddings_size``
        :param log_dir: dir to save models
        :exact_embeddings_size: flag whether to set user/item_num from training log
        """
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.noise_theta = noise_theta
        self.noise_sigma = noise_sigma
        self.noise_type = noise_type
        self.seed = seed
        self.user_num = user_num
        self.item_num = item_num
        self.log_dir = Path(log_dir)
        self.exact_embeddings_size = exact_embeddings_size

    @property
    def _init_args(self):
        return {
            "noise_sigma": self.noise_sigma,
            "noise_theta": self.noise_theta,
            "noise_type": self.noise_type,
            "seed": self.seed,
            "user_num": self.user_num,
            "item_num": self.item_num,
            "log_dir": self.log_dir,
            "exact_embeddings_size": self.exact_embeddings_size,
        }

    # pylint: disable=too-many-locals
    def _batch_pass(self, batch: dict) -> Dict[str, Any]:
        user = torch.FloatTensor(batch["user"])
        memory = torch.FloatTensor(batch["memory"])
        action = torch.FloatTensor(batch["action"])
        reward = torch.FloatTensor(batch["reward"])
        next_user = torch.FloatTensor(batch["next_user"])
        next_memory = torch.FloatTensor(batch["next_memory"])
        done = torch.FloatTensor(batch["done"])

        state = self.model.state_repr(user, memory)
        policy_loss = self.value_net(state, self.model(user, memory))
        policy_loss = -policy_loss.mean()

        next_state = self.model.state_repr(next_user, next_memory)
        next_action = self.target_model(next_user, next_memory)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(
            expected_value, self.min_value, self.max_value
        )

        value = self.value_net(state, action)
        value_loss = (value - expected_value.detach()).squeeze(1).pow(2).mean()

        return policy_loss, value_loss

    @staticmethod
    # pylint: disable=not-callable
    def _predict_pairs_inner(
        model,
        user_idx: int,
        items_np: np.ndarray,
    ) -> DataFrame:
        with torch.no_grad():
            user_batch = torch.LongTensor([user_idx])
            action_emb = model(
                user_batch,
                torch.tensor(model.environment.memory)[
                    to_np(user_batch).astype(int), :
                ],
            )
            user_recs, _ = model.get_action(action_emb, items_np, True)
            user_recs = user_recs.squeeze(1)

            return pd.DataFrame(
                {
                    "user_idx": user_recs.shape[0] * [user_idx],
                    "item_idx": items_np,
                    "relevance": user_recs,
                }
            )

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        items_consider_in_pred = items.toPandas()["item_idx"].values
        model = self.model.cpu()

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            return DDPG._predict_pairs_inner(
                model=model,
                user_idx=pandas_df["user_idx"][0],
                items_np=items_consider_in_pred,
            )[["user_idx", "item_idx", "relevance"]]

        self.logger.debug("Predict started")
        recs = (
            users.join(log, how="left", on="user_idx")
            .select("user_idx", "item_idx")
            .groupby("user_idx")
            .applyInPandas(grouped_map, REC_SCHEMA)
        )
        return recs

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        model = self.model.cpu()

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            return DDPG._predict_pairs_inner(
                model=model,
                user_idx=pandas_df["user_idx"][0],
                items_np=np.array(pandas_df["item_idx_to_pred"][0]),
            )

        self.logger.debug("Calculate relevance for user-item pairs")

        recs = (
            pairs.groupBy("user_idx")
            .agg(sf.collect_list("item_idx").alias("item_idx_to_pred"))
            .join(
                log.select("user_idx").distinct(), on="user_idx", how="inner"
            )
            .groupby("user_idx")
            .applyInPandas(grouped_map, REC_SCHEMA)
        )

        return recs

    @staticmethod
    def _get_beta(idx, beta_start=0.4, beta_steps=100000):
        return min(1.0, beta_start + idx * (1.0 - beta_start) / beta_steps)

    @staticmethod
    def _preprocess_log(log):
        """
        :param log: pyspark DataFrame
        """
        data = log.toPandas()[["user_idx", "item_idx", "relevance"]]
        train_data = data.values.tolist()

        user_num = data["user_idx"].max() + 1
        item_num = data["item_idx"].max() + 1

        train_mat = defaultdict(float)
        for user, item, rel in train_data:
            train_mat[user, item] = rel
        train_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        dict.update(train_matrix, train_mat)

        appropriate_users = data["user_idx"].unique()

        return train_matrix, user_num, item_num, appropriate_users

    def _get_batch(self, step: int = 0) -> dict:
        beta = self._get_beta(step)
        batch = self.replay_buffer.sample(self.batch_size, beta)
        return batch

    def _run_train_step(self, batch: dict) -> None:
        policy_loss, value_loss = self._batch_pass(batch)

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self._target_update(self.target_value_net, self.value_net)
        self._target_update(self.target_model, self.model)

    @staticmethod
    def _target_update(target_net, net, soft_tau=1e-3):
        for target_param, param in zip(
            target_net.parameters(), net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def _init_inner(self):
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.ou_noise = OUNoise(
            self.embedding_dim,
            theta=self.noise_theta,
            max_sigma=self.noise_sigma,
            min_sigma=self.noise_sigma,
            noise_type=self.noise_type,
        )

        self.model = ActorDRR(
            self.user_num,
            self.item_num,
            self.embedding_dim,
            self.hidden_dim,
            self.memory_size,
        )
        self.target_model = ActorDRR(
            self.user_num,
            self.item_num,
            self.embedding_dim,
            self.hidden_dim,
            self.memory_size,
        )
        self.value_net = CriticDRR(
            self.embedding_dim * 3, self.embedding_dim, self.hidden_dim
        )
        self.target_value_net = CriticDRR(
            self.embedding_dim * 3, self.embedding_dim, self.hidden_dim
        )
        self._target_update(self.target_value_net, self.value_net, soft_tau=1)
        self._target_update(self.target_model, self.model, soft_tau=1)

        self.policy_optimizer = Ranger(
            self.model.parameters(),
            lr=self.policy_lr,
            weight_decay=self.policy_decay,
        )
        self.value_optimizer = Ranger(
            self.value_net.parameters(),
            lr=self.value_lr,
            weight_decay=self.value_decay,
        )

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        train_matrix, user_num, item_num, users = self._preprocess_log(log)

        if self.exact_embeddings_size:
            self.user_num = user_num
            self.item_num = item_num
        self._init_inner()

        self.model.environment.update_env(matrix=train_matrix)
        users = np.random.permutation(users)

        self.logger.debug("Training DDPG")
        self.train(users)

    def train(self, users: np.array) -> None:
        """
        Run training loop

        :param users: array with users for training
        :return:
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        step = 0

        for user in tqdm.auto.tqdm(users):
            user, memory = self.model.environment.reset(user)
            self.ou_noise.reset()
            for user_step in range(len(self.model.environment.related_items)):
                action_emb = self.model(user, memory)
                action_emb = self.ou_noise.get_action(action_emb[0], user_step)
                action = self.model.get_action(
                    action_emb,
                    self.model.environment.available_items,
                )
                user, memory, _, _ = self.model.environment.step(
                    action, action_emb, self.replay_buffer
                )

                if len(self.replay_buffer) > self.batch_size:
                    batch = self._get_batch(step)
                    self._run_train_step(batch)

                if step % self.checkpoint_step == 0 and step > 0:
                    self._save_model(self.log_dir / f"model_{step}.pt")
                step += 1

        self._save_model(self.log_dir / "model_final.pt")

    def _save_model(self, path: str) -> None:
        self.logger.debug(
            "-- Saving model to file (user_num=%d, item_num=%d)",
            self.user_num,
            self.item_num,
        )
        torch.save(
            {
                "actor": self.model.state_dict(),
                "critic": self.value_net.state_dict(),
                "memory": self.model.environment.memory,
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
            },
            path,
        )

    def _load_model(self, path: str) -> None:
        self.logger.debug("-- Loading model from file")
        self._init_inner()

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["actor"])
        self.value_net.load_state_dict(checkpoint["critic"])
        self.model.environment.memory = checkpoint["memory"]
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])

        self._target_update(self.target_value_net, self.value_net, soft_tau=1)
        self._target_update(self.target_model, self.model, soft_tau=1)
