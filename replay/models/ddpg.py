import pickle
import tqdm
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.utils.data as td
from pandas import DataFrame
from pytorch_ranger import Ranger

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


class EvalDataset(td.Dataset):
    def __init__(
        self,
        positive_data,
        item_num,
        positive_mat,
        negative_samples=99,
    ):
        super(EvalDataset, self).__init__()
        self.positive_data = np.array(positive_data)
        self.item_num = item_num
        self.positive_mat = positive_mat
        self.negative_samples = negative_samples

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
        for user, positive, rel in self.positive_data:
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
        self.reset()

    def reset(self):
        self.state = np.zeros(self.action_dim)

    def evolve_state(self):
        x = self.state
        dx = -self.theta * x + self.sigma * np.random.randn(self.action_dim)
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
    def __init__(self, user_num, item_num, embedding_dim, hidden_dim, N):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.state_repr = State_Repr_Module(
            user_num, item_num, embedding_dim, N
        )

        self.initialize()

        self.environment = Env(item_num, user_num, N)
        self.test_environment = Env(item_num, user_num, N)

    def initialize(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, user, memory):
        state = self.state_repr(user, memory)
        return self.layers(state)

    def get_action(
        self,
        action_emb,
        items,
        return_scores=False,
    ):
        scores = torch.bmm(
            self.state_repr.item_embeddings(items).unsqueeze(0),
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
    def __init__(self, item_num, user_num, N):
        """
        Memory is initialized as ['item_num'] * 'N' for each user.
        It is padding indexes in state_repr and will result in zero embeddings.
        """
        self.item_count = item_num
        self.user_count = user_num
        self.N = N
        self.memory = np.ones([user_num, N]) * item_num

    def update_env(self, matrix=None, item_count=None, memory=None):
        if item_count is not None:
            self.item_count = item_count
        if matrix is not None:
            self.matrix = matrix.copy()
        if memory is not None:
            self.memory = memory.copy()

    def reset_without_sampling(self, user_id):
        self.user_id = user_id
        self.related_items = list(
            np.argwhere(self.matrix[self.user_id] > 0)[:, 1][: self.item_count]
        )
        self.num_rele = len(self.related_items)
        self.nonrelated_items = list(
            np.argwhere(self.matrix[self.user_id] < 0)[:, 1][: self.item_count]
        )
        self.available_items = self.related_items + self.nonrelated_items

        return torch.tensor([self.user_id]), torch.tensor(
            self.memory[[self.user_id], :]
        )

    def reset(self, user_id):
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

        self.available_items.remove(to_np(action)[0])
        if len(self.available_items) == len(self.related_items):
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
            0,  # done,
        )


class State_Repr_Module(nn.Module):
    def __init__(
        self,
        user_num,
        item_num,
        embedding_dim,
        N,
    ):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(
            item_num + 1, embedding_dim, padding_idx=int(item_num)
        )
        self.drr_ave = torch.nn.Conv1d(
            in_channels=N, out_channels=1, kernel_size=1
        )

        self.initialize()

    def initialize(self):
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
    N: int = 5
    min_value: int = -10
    max_value: int = 10
    buffer_size: int = 1000000
    _search_space = {
        "noise_sigma": {"type": "uniform", "args": [0.1, 0.6]},
        "noise_theta": {"type": "uniform", "args": [0.1, 0.4]},
    }

    def __init__(
        self,
        noise_sigma: float = 0.4,
        noise_theta: float = 0.1,
        seed: int = 16,
        user_num: int = 5000,
        item_num: int = 200000,
        log_dir: str = "logs/tmp",
    ):
        """
        :param noise_sigma: Ornstein-Uhlenbeck noise sigma value
        :param noise_theta: Ornstein-Uhlenbeck noise theta value
        :param seed: random seed
        :param user_num: number of users
        :param item_num: number of items
        :param log_dir: dir to save models
        """
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.noise_theta = noise_theta
        self.noise_sigma = noise_sigma
        self.user_num = user_num
        self.item_num = item_num
        self.log_dir = Path(log_dir)
        self.replay_buffer = Buffer(self.buffer_size)

        self.model = Actor_DRR(
            user_num, item_num, self.embedding_dim, self.hidden_dim, self.N
        )
        self.target_model = Actor_DRR(
            user_num, item_num, self.embedding_dim, self.hidden_dim, self.N
        )
        self.value_net = Critic_DRR(
            self.embedding_dim * 3, self.embedding_dim, self.hidden_dim
        )
        self.target_value_net = Critic_DRR(
            self.embedding_dim * 3, self.embedding_dim, self.hidden_dim
        )
        self._target_update(self.target_value_net, self.value_net, soft_tau=1)
        self._target_update(self.target_model, self.model, soft_tau=1)

    @property
    def _init_args(self):
        return {
            "noise_sigma": self.noise_sigma,
            "noise_theta": self.noise_theta,
            "user_num": self.user_num,
            "item_num": self.item_num,
        }

    def _batch_pass(self, batch, model) -> Dict[str, Any]:
        pass

    def _loss(self, **kwargs) -> torch.Tensor:
        pass

    @staticmethod
    def _predict_pairs_inner(
        model,
        user_idx: int,
        items_np: np.ndarray,
        cnt: Optional[int] = None,
    ) -> DataFrame:
        with torch.no_grad():
            user_batch = torch.LongTensor([user_idx])
            action_emb = model(
                user_batch,
                torch.tensor(model.environment.memory)[to_np(user_batch).astype(int), :],
            )
            user_recs, _ = model.get_action(
                action_emb,
                torch.tensor(items_np).long(),
                return_scores=True,
            )
            user_recs = user_recs.squeeze(1)

            if cnt is not None:
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

    @staticmethod
    def _predict_by_user(
        pandas_df: pd.DataFrame,
        model: nn.Module,
        items_np: np.ndarray,
        k: int,
        item_count: int,
    ) -> pd.DataFrame:
        return DDPG._predict_pairs_inner(
            model=model,
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
        print(pandas_df, item_count)
        return DDPG._predict_pairs_inner(
            model=model,
            user_idx=pandas_df["user_idx"][0],
            items_np=np.array(pandas_df["item_idx_to_pred"][0]),
            cnt=None,
        )

    def _get_data_loader(self, data, item_num, matrix):
        dataset = EvalDataset(data, item_num, matrix)
        loader = td.DataLoader(
            dataset, batch_size=100, shuffle=False, num_workers=16
        )
        return loader

    def _get_beta(self, idx, beta_start=0.4, beta_steps=100000):
        return min(1.0, beta_start + idx * (1.0 - beta_start) / beta_steps)

    def _preprocess_log(self, log):
        data = log.toPandas()[["user_idx", "item_idx", "relevance"]]
        user_num = data["user_idx"].max() + 1
        item_num = data["item_idx"].max() + 1

        train_data = data.sample(frac=0.9, random_state=16)
        appropriate_users = (
            train_data["user_idx"]
            .value_counts()[train_data["user_idx"].value_counts() > 10]
            .index
        )
        test_data = data.drop(train_data.index).values.tolist()
        train_data = train_data.values.tolist()

        train_mat = defaultdict(float)
        test_mat = defaultdict(float)
        for user, item, rel in train_data:
            train_mat[user, item] = rel
        for user, item, rel in test_data:
            test_mat[user, item] = rel
        train_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        dict.update(train_matrix, train_mat)
        test_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        dict.update(test_matrix, test_mat)

        return (
            train_matrix,
            test_data,
            test_matrix,
            item_num,
            appropriate_users,
        )

    def _hit_metric(self, recommended, actual):
        return int(actual in recommended)

    def _dcg_metric(self, recommended, actual):
        if actual in recommended:
            index = recommended.index(actual)
            return np.reciprocal(np.log2(index + 2))
        return 0

    def _run_train_step(
        self,
        policy_optimizer,
        value_optimizer,
        step=0,
    ):
        beta = self._get_beta(step)
        batch = self.replay_buffer.sample(self.batch_size, beta)
        user = torch.FloatTensor(batch[0])
        memory = torch.FloatTensor(batch[1])
        action = torch.FloatTensor(batch[2])
        reward = torch.FloatTensor(batch[3])
        next_user = torch.FloatTensor(batch[4])
        next_memory = torch.FloatTensor(batch[5])
        done = torch.FloatTensor(batch[6])

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

        policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        policy_optimizer.step()
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        self._target_update(self.target_value_net, self.value_net)
        self._target_update(self.target_model, self.model)

    def _target_update(self, target_net, net, soft_tau=1e-3):
        for target_param, param in zip(
            target_net.parameters(), net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def _run_validation(self, loader):
        self.model.eval()
        hits = []
        dcgs = []
        with torch.no_grad():
            self.model.test_environment.update_env(
                memory=self.model.environment.memory
            )
            user, memory = self.model.test_environment.reset(
                int(to_np(next(iter(loader))["user"])[0])
            )
            for batch in loader:
                action_emb = self.model(user, memory)
                scores, action = self.model.get_action(
                    action_emb,
                    batch["item"].long(),
                    return_scores=True,
                )
                user, memory, _, _ = self.model.test_environment.step(action)

                _, ind = scores[:, 0].topk(10)
                predictions = batch["item"].take(ind).cpu().numpy().tolist()
                actual = batch["item"][0].item()
                hits.append(self._hit_metric(predictions, actual))
                dcgs.append(self._dcg_metric(predictions, actual))
        return np.mean(hits), np.mean(dcgs)

    def load_user_embeddings(self, user_embeddings_path):
        user_embeddings = pd.read_parquet(user_embeddings_path)
        user_embeddings = user_embeddings[
            user_embeddings["user_idx"] < self.user_num
        ]
        indexes = user_embeddings["user_idx"]
        embeddings = torch.from_numpy(
            user_embeddings.iloc[:, -8:].values
        ).float()
        self.model.state_repr.user_embeddings.weight.data[indexes] = embeddings

    def load_item_embeddings(self, item_embeddings_path):
        item_embeddings = pd.read_parquet(item_embeddings_path)
        item_embeddings = item_embeddings[
            item_embeddings["item_idx"] < self.item_num
        ]
        indexes = item_embeddings["item_idx"]
        embeddings = torch.from_numpy(
            item_embeddings.iloc[:, -8:].values
        ).float()
        self.model.state_repr.item_embeddings.weight.data[indexes] = embeddings

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
            current_item_num,
            appropriate_users,
        ) = self._preprocess_log(log)
        self.model.environment.update_env(
            matrix=train_matrix  # , item_count=current_item_num
        )
        self.model.test_environment.update_env(
            matrix=test_matrix  # , item_count=current_item_num
        )
        users = np.random.permutation(appropriate_users)
        # valid_loader = self._get_data_loader(
        #     np.array(test_data)[np.array(test_data)[:, 0] == 16],
        #     current_item_num,
        #     test_matrix,
        # )

        self.ou_noise = OUNoise(
            self.embedding_dim,
            theta=self.noise_theta,
            max_sigma=self.noise_sigma,
            min_sigma=self.noise_sigma,
        )
        policy_optimizer = Ranger(
            self.model.parameters(),
            lr=self.policy_lr,
            weight_decay=self.policy_decay,
        )
        value_optimizer = Ranger(
            self.value_net.parameters(),
            lr=self.value_lr,
            weight_decay=self.value_decay,
        )

        self.logger.debug("Training DDPG")
        self.train(
            policy_optimizer,
            value_optimizer,
            users,
            valid_loader=None,
        )

    def train(
        self,
        policy_optimizer,
        value_optimizer,
        users,
        valid_loader=None,
    ):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        hits, dcgs, rewards = [], [], []
        step, best_step = 0, 0

        for i, u in enumerate(tqdm.tqdm(users)):
            user, memory = self.model.environment.reset(u)
            self.ou_noise.reset()
            for t in range(len(self.model.environment.related_items)):
                action_emb = self.model(user, memory)
                action_emb = self.ou_noise.get_action(to_np(action_emb)[0], t)
                action = self.model.get_action(
                    action_emb,
                    torch.tensor(self.model.environment.available_items).long(),
                )
                user, memory, reward, _ = self.model.environment.step(
                    action, action_emb, self.replay_buffer
                )
                rewards.append(reward)

                if len(self.replay_buffer) > self.batch_size:
                    self._run_train_step(
                        policy_optimizer,
                        value_optimizer,
                        step=step,
                    )

                if step % 10000 == 0 and step > 0:
                    self._save_model(self.log_dir / f"model_{step}.pt")
                step += 1

        self._save_model(self.log_dir / "model_final.pt")
        self._save_memory()

    def _save_memory(self) -> None:
        with open(self.log_dir / "memory.pickle", "wb") as f:
            pickle.dump(self.model.environment.memory, f)

    def _load_memory(self, path: str = ''):
        with open(path, 'rb') as f:
            self.model.environment.memory = pickle.load(f)
