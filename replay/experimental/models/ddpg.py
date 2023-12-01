# pylint: disable=too-many-lines
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch
import tqdm
from pytorch_ranger import Ranger
from torch import nn
from torch.distributions.gamma import Gamma

from replay.data import get_schema
from replay.experimental.models.base_torch_rec import Recommender
from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, SparkDataFrame
from replay.utils.spark_utils import convert2spark

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf


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
    def __init__(self, device, capacity, memory_size, embedding_dim):
        self.capacity = capacity

        self.buffer = {
            "user": torch.zeros((capacity,), device=device),
            "memory": torch.zeros((capacity, memory_size), device=device),
            "action": torch.zeros((capacity, embedding_dim), device=device),
            "reward": torch.zeros((capacity,), device=device),
            "next_user": torch.zeros((capacity,), device=device),
            "next_memory": torch.zeros((capacity, memory_size), device=device),
            "done": torch.zeros((capacity,), device=device),
            "sample_weight": torch.zeros((capacity,), device=device),
        }

        self.pos = 0
        self.is_filled = False

    def push(
        self,
        user,
        memory,
        action,
        reward,
        next_user,
        next_memory,
        done,
        sample_weight,
    ):
        """Add transition to buffer."""

        batch_size = user.shape[0]

        self.buffer["user"][self.pos : self.pos + batch_size] = user
        self.buffer["memory"][self.pos : self.pos + batch_size] = memory
        self.buffer["action"][self.pos : self.pos + batch_size] = action
        self.buffer["reward"][self.pos : self.pos + batch_size] = reward
        self.buffer["next_user"][self.pos : self.pos + batch_size] = next_user
        self.buffer["next_memory"][
            self.pos : self.pos + batch_size
        ] = next_memory
        self.buffer["done"][self.pos : self.pos + batch_size] = done
        self.buffer["sample_weight"][
            self.pos : self.pos + batch_size
        ] = sample_weight

        new_pos = self.pos + batch_size
        if new_pos >= self.capacity:
            self.is_filled = True
        self.pos = new_pos % self.capacity

    # pylint: disable=too-many-locals
    def sample(self, batch_size):
        """Sample transition from buffer."""
        current_buffer_len = len(self)

        indices = np.random.choice(current_buffer_len, batch_size)

        return {
            "user": self.buffer["user"][indices],
            "memory": self.buffer["memory"][indices],
            "action": self.buffer["action"][indices],
            "reward": self.buffer["reward"][indices],
            "next_user": self.buffer["next_user"][indices],
            "next_memory": self.buffer["next_memory"][indices],
            "done": self.buffer["done"][indices],
            "sample_weight": self.buffer["sample_weight"][indices],
        }

    def __len__(self):
        return self.capacity if self.is_filled else self.pos + 1


# pylint: disable=too-many-instance-attributes,too-many-arguments,not-callable
class OUNoise:
    """https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py"""

    def __init__(
        self,
        action_dim,
        device,
        theta=0.15,
        max_sigma=0.4,
        min_sigma=0.4,
        noise_type="gauss",
        decay_period=10,
    ):
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.noise_type = noise_type
        self.device = device
        self.state = torch.zeros((1, action_dim), device=self.device)

    def reset(self, user_batch_size):
        """Fill state with zeros."""
        if self.state.shape[0] == user_batch_size:
            self.state.fill_(0)
        else:
            self.state = torch.zeros(
                (user_batch_size, self.action_dim), device=self.device
            )

    def evolve_state(self):
        """Perform OU discrete approximation step"""
        x = self.state
        d_x = -self.theta * x + self.sigma * torch.randn(
            x.shape, device=self.device
        )
        self.state = x + d_x
        return self.state

    def get_action(self, action, step=0):
        """Get state after applying noise."""
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, step / self.decay_period
        )
        if self.noise_type == "ou":
            ou_state = self.evolve_state()
            return action + ou_state
        elif self.noise_type == "gauss":
            return action + self.sigma * torch.randn(
                action.shape, device=self.device
            )
        else:
            raise ValueError("noise_type must be one of ['ou', 'gauss']")


class ActorDRR(nn.Module):
    """
    DDPG Actor model (based on `DRR
    <https://arxiv.org/pdf/1802.05814.pdf>`).
    """

    def __init__(
        self,
        user_num,
        item_num,
        embedding_dim,
        hidden_dim,
        memory_size,
        env_gamma_alpha,
        device,
        min_trajectory_len,
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

        self.environment = Env(
            item_num,
            user_num,
            memory_size,
            env_gamma_alpha,
            device,
            min_trajectory_len,
        )

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
    def get_action(self, action_emb, items, items_mask, return_scores=False):
        """
        :param action_emb: output of the .forward() (user_batch_size x emb_dim)
        :param items: items batch (user_batch_size x items_num)
        :param items_mask: mask of available items for reccomendation (user_batch_size x items_num)
        :param return_scores: whether to return scores of items
        :return: output, prediction (and scores if return_scores)
        """

        assert items.shape == items_mask.shape

        items = self.state_repr.item_embeddings(items)  # B x i x emb_dim
        scores = torch.bmm(
            items,
            action_emb.unsqueeze(-1),  # B x emb_dim x 1
        ).squeeze(-1)

        assert scores.shape == items_mask.shape

        scores *= items_mask

        if return_scores:
            return scores, torch.argmax(scores, dim=1)
        else:
            return torch.argmax(scores, dim=1)


class CriticDRR(nn.Module):
    """
    DDPG Critic model (based on `DRR
    <https://arxiv.org/pdf/1802.05814.pdf>`
    and `Bayes-UCBDQN <https://arxiv.org/pdf/2205.07704.pdf>`).
    """

    def __init__(
        self, state_repr_dim, action_emb_dim, hidden_dim, heads_num, heads_q
    ):
        """
        :param heads_num: number of heads (samples of Q funtion)
        :param heads_q: quantile of Q function distribution
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_repr_dim + action_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(heads_num)]
        )
        self.heads_q = heads_q

        self.initialize()

    def initialize(self):
        """weight init"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

        for head in self.heads:
            nn.init.kaiming_uniform_(head.weight)

    def forward(self, state, action):
        """
        :param state: state batch
        :param action: action batch
        :return: x, Q values for given states and actions
        """
        x = torch.cat([state, action], 1)
        out = self.layers(x)
        heads_out = torch.stack([head(out) for head in self.heads])
        out = torch.quantile(heads_out, self.heads_q, dim=0)

        return out


# pylint: disable=too-many-instance-attributes, not-callable
class Env:
    """
    RL environment for recommender systems.
    Simulates interacting with a batch of users

    Keep users' latest relevant items (memory).

    :param item_count: total number of items
    :param user_count: total number of users
    :param memory_size: maximum number of items in memory
    :param memory: torch.tensor with users' latest relevant items
    :param matrix: sparse matrix with users-item ratings
    :param user_ids: user ids from the batch
    :param related_items: relevant items for current users
    :param nonrelated_items: non-relevant items for current users
    :param max_num_rele: maximum number of related items by users in the batch
    :param available_items: items available for recommendation
    :param available_items_mask: mask of non-seen items
    :param gamma: param of Gamma distibution for sample weights
    """

    matrix: np.array
    related_items: torch.Tensor
    nonrelated_items: torch.Tensor
    available_items: torch.Tensor  # B x i
    available_items_mask: torch.Tensor  # B x i
    user_id: torch.Tensor  # batch of users B x i
    num_rele: int

    def __init__(
        self,
        item_count,
        user_count,
        memory_size,
        gamma_alpha,
        device,
        min_trajectory_len,
    ):
        """
        Initialize memory as ['item_num'] * 'memory_size' for each user.

        'item_num' is a padding index in StateReprModule.
        It will result in zero embeddings.
        """
        self.item_count = item_count
        self.user_count = user_count
        self.memory_size = memory_size
        self.device = device
        self.gamma = Gamma(
            torch.tensor([float(gamma_alpha)]),
            torch.tensor([1 / float(gamma_alpha)]),
        )
        self.memory = torch.full(
            (user_count, memory_size), item_count, device=device
        )
        self.min_trajectory_len = min_trajectory_len
        self.max_num_rele = None
        self.user_batch_size = None
        self.user_ids = None

    def update_env(self, matrix=None, item_count=None):
        """Update some of Env attributes."""
        if item_count is not None:
            self.item_count = item_count
        if matrix is not None:
            self.matrix = matrix.copy()

    def reset(self, user_ids):
        """
        :param user_id: batch of user ids
        :return: user, memory
        """
        self.user_batch_size = len(user_ids)

        self.user_ids = torch.tensor(
            user_ids, dtype=torch.int64, device=self.device
        )

        self.max_num_rele = max(
            (self.matrix[user_ids] > 0).sum(1).max(), self.min_trajectory_len
        )
        self.available_items = torch.zeros(
            (self.user_batch_size, 2 * self.max_num_rele),
            dtype=torch.int64,
            device=self.device,
        )
        self.available_items_mask = torch.ones_like(
            self.available_items, device=self.device
        )

        # padding with non-existent items
        self.related_items = torch.full(
            (self.user_batch_size, self.max_num_rele),
            -1,  # maybe define new constant
            device=self.device,
        )

        for idx, user_id in enumerate(user_ids):
            user_related_items = torch.tensor(
                np.argwhere(self.matrix[user_id] > 0)[:, 1], device=self.device
            )

            user_num_rele = len(user_related_items)

            self.related_items[idx, :user_num_rele] = user_related_items

            replace = bool(2 * self.max_num_rele > self.item_count)

            nonrelated_items = torch.tensor(
                np.random.choice(
                    list(
                        set(range(self.item_count + 1))
                        - set(user_related_items.tolist())
                    ),
                    replace=replace,
                    size=2 * self.max_num_rele - user_num_rele,
                )
            ).to(self.device)

            self.available_items[idx, :user_num_rele] = user_related_items
            self.available_items[idx, user_num_rele:] = nonrelated_items
            self.available_items[self.available_items == -1] = self.item_count
            perm = torch.randperm(self.available_items.shape[1])
            self.available_items[idx] = self.available_items[idx, perm]

        return self.user_ids, self.memory[self.user_ids]

    def step(self, actions, actions_emb=None, buffer: ReplayBuffer = None):
        """Execute step and return (user, memory) for new state"""
        initial_users = self.user_ids
        initial_memory = self.memory[self.user_ids].clone()

        global_actions = self.available_items[
            torch.arange(self.available_items.shape[0]), actions
        ]
        rewards = (global_actions.reshape(-1, 1) == self.related_items).sum(1)
        for idx, reward in enumerate(rewards):
            if reward:
                user_id = self.user_ids[idx]
                self.memory[user_id] = torch.tensor(
                    list(self.memory[user_id][1:]) + [global_actions[idx]]
                )

        self.available_items_mask[
            torch.arange(self.available_items_mask.shape[0]), actions
        ] = 0

        if buffer is not None:
            sample_weight = (
                self.gamma.sample((self.user_batch_size,))
                .squeeze()
                .detach()
                .to(self.device)
            )
            buffer.push(
                initial_users.detach(),
                initial_memory.detach(),
                actions_emb.detach(),
                rewards.detach(),
                self.user_ids.detach(),
                self.memory[self.user_ids].detach(),
                rewards.detach(),
                sample_weight,
            )

        return self.user_ids, self.memory[self.user_ids], rewards, 0


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
        "gamma": {"type": "uniform", "args": [0.7, 1.0]},
        "value_lr": {"type": "loguniform", "args": [1e-7, 1e-1]},
        "value_decay": {"type": "loguniform", "args": [1e-7, 1e-1]},
        "policy_lr": {"type": "loguniform", "args": [1e-7, 1e-1]},
        "policy_decay": {"type": "loguniform", "args": [1e-7, 1e-1]},
        "memory_size": {"type": "categorical", "args": [3, 5, 7, 9]},
        "noise_type": {"type": "categorical", "args": ["gauss", "ou"]},
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
    # pylint: disable=too-many-locals
    def __init__(
        self,
        noise_sigma: float = 0.2,
        noise_theta: float = 0.05,
        noise_type: str = "gauss",
        seed: int = 9,
        user_num: int = 10,
        item_num: int = 10,
        log_dir: str = "logs/tmp",
        exact_embeddings_size=True,
        n_critics_head: int = 10,
        env_gamma_alpha: float = 0.2,
        critic_heads_q: float = 0.15,
        n_jobs=None,
        use_gpu=False,
        user_batch_size: int = 8,
        min_trajectory_len: int = 10,
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
        self.n_critics_head = n_critics_head
        self.env_gamma_alpha = env_gamma_alpha
        self.critic_heads_q = critic_heads_q
        self.user_batch_size = user_batch_size
        self.min_trajectory_len = min_trajectory_len
        if n_jobs is not None:
            torch.set_num_threads(n_jobs)

        if use_gpu:
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

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
        user = batch["user"]
        memory = batch["memory"]
        action = batch["action"]
        reward = batch["reward"]
        next_user = batch["next_user"]
        next_memory = batch["next_memory"]
        done = batch["done"]
        sample_weight = batch["sample_weight"]

        state = self.model.state_repr(user, memory)
        policy_loss = self.value_net(state, self.model(user, memory))
        policy_loss = -policy_loss.mean()

        next_state = self.model.state_repr(next_user, next_memory)
        next_action = self.target_model(next_user, next_memory)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (
            1.0 - done
        ) * self.gamma * target_value.squeeze(
            1
        )  # smth strange, check article

        expected_value = torch.clamp(
            expected_value, self.min_value, self.max_value
        )

        value = self.value_net(state, action)
        value_loss = (
            (((value - expected_value.detach())).pow(2) * sample_weight)
            .squeeze(1)
            .mean()
        )

        return policy_loss, value_loss

    @staticmethod
    # pylint: disable=not-callable
    def _predict_pairs_inner(
        model,
        user_idx: int,
        items_np: np.ndarray,
    ) -> SparkDataFrame:
        with torch.no_grad():
            # user_batch, memory = model.environment.reset([user_idx])
            user_batch = torch.tensor([user_idx], dtype=torch.int64)
            memory = model.environment.memory[user_batch]
            action_emb = model(user_batch, memory)
            items = torch.tensor(items_np, dtype=torch.int64).unsqueeze(0)
            scores, _ = model.get_action(
                action_emb, items, torch.full_like(items, True), True
            )
            scores = scores.squeeze()
            return PandasDataFrame(
                {
                    "user_idx": scores.shape[0] * [user_idx],
                    "item_idx": items_np,
                    "relevance": scores,
                }
            )

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: SparkDataFrame,
        k: int,
        users: SparkDataFrame,
        items: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
        filter_seen_items: bool = True,
    ) -> SparkDataFrame:
        items_consider_in_pred = items.toPandas()["item_idx"].values
        model = self.model.cpu()

        def grouped_map(pandas_df: PandasDataFrame) -> PandasDataFrame:
            return DDPG._predict_pairs_inner(
                model=model,
                user_idx=pandas_df["user_idx"][0],
                items_np=items_consider_in_pred,
            )[["user_idx", "item_idx", "relevance"]]

        self.logger.debug("Predict started")
        rec_schema = get_schema(
            query_column="user_idx",
            item_column="item_idx",
            rating_column="relevance",
            has_timestamp=False,
        )
        recs = (
            users.join(log, how="left", on="user_idx")
            .select("user_idx", "item_idx")
            .groupby("user_idx")
            .applyInPandas(grouped_map, rec_schema)
        )
        return recs

    def _predict_pairs(
        self,
        pairs: SparkDataFrame,
        log: Optional[SparkDataFrame] = None,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
    ) -> SparkDataFrame:
        model = self.model.cpu()

        def grouped_map(pandas_df: PandasDataFrame) -> PandasDataFrame:
            return DDPG._predict_pairs_inner(
                model=model,
                user_idx=pandas_df["user_idx"][0],
                items_np=np.array(pandas_df["item_idx_to_pred"][0]),
            )

        self.logger.debug("Calculate relevance for user-item pairs")

        rec_schema = get_schema(
            query_column="user_idx",
            item_column="item_idx",
            rating_column="relevance",
            has_timestamp=False,
        )
        recs = (
            pairs.groupBy("user_idx")
            .agg(sf.collect_list("item_idx").alias("item_idx_to_pred"))
            .join(
                log.select("user_idx").distinct(), on="user_idx", how="inner"
            )
            .groupby("user_idx")
            .applyInPandas(grouped_map, rec_schema)
        )

        return recs

    @staticmethod
    def _preprocess_df(data):
        """
        :param data: pandas DataFrame
        """
        data = data[["user_idx", "item_idx", "relevance"]]
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

    @staticmethod
    def _preprocess_log(log):
        return DDPG._preprocess_df(log.toPandas())

    def _get_batch(self) -> dict:
        batch = self.replay_buffer.sample(self.batch_size)
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
        self.replay_buffer = ReplayBuffer(
            self.device,
            self.buffer_size,
            memory_size=self.memory_size,
            embedding_dim=self.embedding_dim,
        )

        self.ou_noise = OUNoise(
            self.embedding_dim,
            device=self.device,
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
            env_gamma_alpha=self.env_gamma_alpha,
            device=self.device,
            min_trajectory_len=self.min_trajectory_len,
        ).to(self.device)

        self.target_model = ActorDRR(
            self.user_num,
            self.item_num,
            self.embedding_dim,
            self.hidden_dim,
            self.memory_size,
            env_gamma_alpha=self.env_gamma_alpha,
            device=self.device,
            min_trajectory_len=self.min_trajectory_len,
        ).to(self.device)

        self.value_net = CriticDRR(
            self.embedding_dim * 3,
            self.embedding_dim,
            self.hidden_dim,
            heads_num=self.n_critics_head,
            heads_q=self.critic_heads_q,
        ).to(self.device)

        self.target_value_net = CriticDRR(
            self.embedding_dim * 3,
            self.embedding_dim,
            self.hidden_dim,
            heads_num=self.n_critics_head,
            heads_q=self.critic_heads_q,
        ).to(self.device)

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
        log: SparkDataFrame,
        user_features: Optional[SparkDataFrame] = None,
        item_features: Optional[SparkDataFrame] = None,
    ) -> None:
        data = log.toPandas()
        self._fit_df(data)

    def _fit_df(self, data):
        train_matrix, user_num, item_num, users = self._preprocess_df(data)

        if self.exact_embeddings_size:
            self.user_num = user_num
            self.item_num = item_num
        self._init_inner()

        self.model.environment.update_env(matrix=train_matrix)
        users = np.random.permutation(users)

        self.logger.debug("Training DDPG")
        self.train(users)

    @staticmethod
    def users_loader(users, batch_size):
        """loader for users' batch"""
        pos = 0
        while pos != len(users):
            new_pos = min(pos + batch_size, len(users))
            yield users[pos:new_pos]
            pos = new_pos

    def train(self, users: np.array) -> None:
        """
        Run training loop

        :param users: array with users for training
        :return:
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        step = 0
        users_loader = self.users_loader(users, self.user_batch_size)
        for user_ids in tqdm.auto.tqdm(list(users_loader)):
            user_ids, memory = self.model.environment.reset(user_ids)
            self.ou_noise.reset(user_ids.shape[0])
            for users_step in range(self.model.environment.max_num_rele):
                actions_emb = self.model(user_ids, memory)
                actions_emb = self.ou_noise.get_action(actions_emb, users_step)

                actions = self.model.get_action(
                    actions_emb,
                    self.model.environment.available_items,
                    self.model.environment.available_items_mask,
                )

                _, memory, _, _ = self.model.environment.step(
                    actions, actions_emb, self.replay_buffer
                )

                if len(self.replay_buffer) > self.batch_size:
                    batch = self._get_batch()
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

        fit_users = getattr(self, "fit_users", None)
        fit_items = getattr(self, "fit_items", None)

        torch.save(
            {
                # pylint: disable-next=used-before-assignment
                "fit_users": fit_users.toPandas()
                if fit_users is not None
                else None,
                # pylint: disable-next=used-before-assignment
                "fit_items": fit_items.toPandas()
                if fit_items is not None
                else None,
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
        self.fit_users = convert2spark(checkpoint["fit_users"])
        self.fit_items = convert2spark(checkpoint["fit_items"])

        self._target_update(self.target_value_net, self.value_net, soft_tau=1)
        self._target_update(self.target_model, self.model, soft_tau=1)
