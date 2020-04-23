"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, Optional, List

from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Engine, Events
from ignite.handlers import (EarlyStopping, ModelCheckpoint,
                             global_step_from_engine)
from ignite.metrics import Loss
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from sklearn.model_selection import train_test_split
import torch.optim
from torch import LongTensor, Tensor
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from sponge_bob_magic.models import TorchRecommender
from sponge_bob_magic.session_handler import State


def xavier_init_(layer: nn.Module):
    """
    Инициализация весов линейного слоя методом Хавьера

    :param layer: слой нейронной сети
    """
    if isinstance(layer, (nn.Embedding, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)

    if isinstance(layer, nn.Linear):
        layer.bias.data.normal_(0.0, 0.001)


class GMF(nn.Module):
    """Generalized Matrix Factorization (GMF) модель - нейросетевая
    реализация матричной факторизации"""
    def __init__(
            self,
            user_count: int,
            item_count: int,
            embedding_dim: int
    ):
        """
        Инициализация модели. Создает эмбеддинги пользователей и объектов.

        :param user_count: количество пользователей
        :param item_count: количество объектов
        :param embedding_dim: размерность представления пользователей и
            объектов
        """
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=user_count,
                                           embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=item_count,
                                           embedding_dim=embedding_dim)
        self.item_biases = nn.Embedding(num_embeddings=item_count,
                                        embedding_dim=1)
        self.user_biases = nn.Embedding(num_embeddings=user_count,
                                        embedding_dim=1)

        xavier_init_(self.user_embedding)
        xavier_init_(self.item_embedding)
        self.user_biases.weight.data.zero_()
        self.item_biases.weight.data.zero_()

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Один проход нейросети.

        :param user: батч ID пользователей
        :param item: батч ID объектов
        :return: батч весов предсказанных взаимодействий пользователей и
            объектов
        """
        user_emb = self.user_embedding(user) + self.user_biases(user)
        item_emb = self.item_embedding(item) + self.item_biases(item)
        element_product = torch.mul(user_emb, item_emb)

        return element_product


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) модель"""
    def __init__(
            self,
            user_count: int,
            item_count: int,
            embedding_dim: int,
            hidden_dims: Optional[List[int]] = None,
    ):
        """
        Инициализация модели.

        :param user_count: количество пользователей
        :param item_count: количество объектов
        :param embedding_dim: размерность представления пользователей и
            объектов
        :param hidden_dims: последовательность размеров скрытых слоев
        """
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=user_count,
                                           embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=item_count,
                                           embedding_dim=embedding_dim)
        self.item_biases = nn.Embedding(num_embeddings=item_count,
                                        embedding_dim=1)
        self.user_biases = nn.Embedding(num_embeddings=user_count,
                                        embedding_dim=1)

        if hidden_dims:
            full_hidden_dims = [2 * embedding_dim] + hidden_dims
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(d_in, d_out) for d_in, d_out in
                 zip(full_hidden_dims[:-1], full_hidden_dims[1:])])

        else:
            self.hidden_layers = nn.ModuleList()

        self.activation = nn.ReLU()

        xavier_init_(self.user_embedding)
        xavier_init_(self.item_embedding)
        self.user_biases.weight.data.zero_()
        self.item_biases.weight.data.zero_()
        for layer in self.hidden_layers:
            xavier_init_(layer)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Один проход нейросети.

        :param user: батч ID пользователей
        :param item: батч ID объектов
        :return: батч весов предсказанных взаимодействий пользователей и
            объектов
        """
        user_emb = self.user_embedding(user) + self.user_biases(user)
        item_emb = self.item_embedding(item) + self.item_biases(item)
        hidden = torch.cat([user_emb, item_emb], dim=-1)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
            hidden = self.activation(hidden)
        return hidden


class NMF(nn.Module):
    """NMF модель (MLP + GMF)"""
    def __init__(
            self,
            user_count: int,
            item_count: int,
            embedding_gmf_dim: Optional[int] = None,
            embedding_mlp_dim: Optional[int] = None,
            hidden_mlp_dims: Optional[List[int]] = None,
    ):
        """
        Инициализация модели. Создает эмбеддинги пользователей и объектов.

        :param user_count: количество пользователей
        :param item_count: количество объектов
        :param embedding_gmf_dim: размерность представления пользователей и
            объектов в модели gmf
        :param embedding_mlp_dim: размерность представления пользователей и
            объектов в модели mlp
        :param hidden_mlp_dims: последовательность размеров скрытых слоев в
            модели mlp
        """
        super().__init__()
        merged_dim = 0
        if embedding_gmf_dim:
            self.gmf = GMF(user_count, item_count, embedding_gmf_dim)
            merged_dim += embedding_gmf_dim
        else:
            self.gmf = None

        if embedding_mlp_dim:
            self.mlp = MLP(user_count, item_count,
                           embedding_mlp_dim, hidden_mlp_dims)
            merged_dim += hidden_mlp_dims[-1]
        else:
            self.mlp = None

        self.last_layer = nn.Linear(merged_dim, 1)
        xavier_init_(self.last_layer)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Один проход нейросети.

        :param user: батч ID пользователей
        :param item: батч ID объектов
        :return: батч весов предсказанных взаимодействий пользователей и
            объектов или батч промежуточных эмбеддингов
        """
        batch_size = len(user)
        if self.gmf:
            gmf_vector = self.gmf(user, item)
        else:
            gmf_vector = torch.zeros(batch_size, 0).to(user.device)

        if self.mlp:
            mlp_vector = self.mlp(user, item)
        else:
            mlp_vector = torch.zeros(batch_size, 0).to(user.device)

        merged_vector = torch.cat([gmf_vector, mlp_vector], dim=1)
        merged_vector = self.last_layer(merged_vector).squeeze()
        merged_vector = torch.sigmoid(merged_vector)

        return merged_vector


class NeuroMF(TorchRecommender):
    """
    Эта модель является вариацей на модель из статьи Neural Matrix Factorization
    (NeuMF, NCF).

    Модель позволяет использовать архитектуры MLP и GMF как отдельно,
    так и совместно.
    """
    num_workers: int = 16
    batch_size_users: int = 100000
    trainer: Engine
    val_evaluator: Engine
    patience: int = 3
    n_saved: int = 2
    valid_split_size: float = 0.1
    seed: int = 42

    def __init__(self, learning_rate: float = 0.05,
                 epochs: int = 5,
                 embedding_gmf_dim: Optional[int] = None,
                 embedding_mlp_dim: Optional[int] = None,
                 hidden_mlp_dims: Optional[List[int]] = None,
                 l2_reg: float = 0,
                 gamma: float = 0.99,
                 count_negative_sample: int = 1
                 ):
        """
        Инициализирует параметры модели.

        :param learning_rate: шаг обучения
        :param epochs: количество эпох, в течение которых учимся
        :param embedding_gmf_dim: размерность представления пользователей и
            объектов в модели gmf
        :param embedding_mlp_dim: размерность представления пользователей и
            объектов в модели mlp
        :param hidden_mlp_dims: последовательность размеров скрытых слоев в
            модели mlp
        :param l2_reg: коэффициент l2 регуляризации
        :param gamma: коэффициент уменьшения learning_rate после каждой эпохи
        :param count_negative_sample: количество отрицательных примеров
        """
        if not embedding_gmf_dim and not embedding_mlp_dim:
            raise ValueError("Хотя бы один из параметров embedding_gmf_dim, "
                             "embedding_mlp_dim должен быть не пуст")

        if ((embedding_gmf_dim is None or embedding_gmf_dim < 0) and
           (embedding_mlp_dim is None or embedding_mlp_dim < 0)):
            raise ValueError("Параметры embedding_gmf_dim, embedding_mlp_dim"
                             " должны быть положительными")

        self.device = State().device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.embedding_gmf_dim = embedding_gmf_dim
        self.embedding_mlp_dim = embedding_mlp_dim
        self.hidden_mlp_dims = hidden_mlp_dims
        self.l2_reg = l2_reg
        self.gamma = gamma
        self.count_negative_sample = count_negative_sample

    def get_params(self) -> Dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs
        }

    def _data_loader(self,
                     data: pd.DataFrame,
                     shuffle: bool = True
                     ) -> DataLoader:

        user_batch = LongTensor(data["user_idx"].values)
        item_batch = LongTensor(data["item_idx"].values)

        dataset = TensorDataset(user_batch, item_batch)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size_users,
            shuffle=shuffle,
            num_workers=self.num_workers
        )
        return loader

    def _get_neg_batch(self, batch: Tensor) -> Tensor:
        negative_items = (torch
                          .randint(0, self.items_count - 1,
                                   (batch.shape[0] *
                                    self.count_negative_sample,)))
        return negative_items

    def _fit(self,
             log: DataFrame,
             user_features: Optional[DataFrame] = None,
             item_features: Optional[DataFrame] = None) -> None:
        self.model = NMF(
            user_count=self.users_count,
            item_count=self.items_count,
            embedding_gmf_dim=self.embedding_gmf_dim,
            embedding_mlp_dim=self.embedding_mlp_dim,
            hidden_mlp_dims=self.hidden_mlp_dims,
        ).to(self.device)

        self.logger.debug("Индексирование данных")
        log_indexed = self.user_indexer.transform(log)
        log_indexed = self.item_indexer.transform(log_indexed)

        self.logger.debug("Составление батча")
        tensor_data = log_indexed.select("user_idx", "item_idx").toPandas()
        train_tensor_data, valid_tensor_data = train_test_split(
            tensor_data,
            stratify=tensor_data["user_idx"],
            test_size=self.valid_split_size,
            random_state=self.seed)
        train_data_loader = self._data_loader(train_tensor_data)
        val_data_loader = self._data_loader(valid_tensor_data)

        self.logger.debug("Обучение модели")
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg / self.batch_size_users)
        lr_scheduler = ExponentialLR(optimizer, gamma=self.gamma)
        scheduler = LRScheduler(lr_scheduler)
        y_true = torch.ones(self.batch_size_users *
                            (1 + self.count_negative_sample)).to(self.device)
        y_true[self.batch_size_users:] = 0

        def _loss(y_pred, y_true):
            if not y_true.shape == y_pred.shape:
                pos_len = y_pred.shape[0] // (self.count_negative_sample + 1)
                y_real_true = torch.cat((
                    y_true[:pos_len],
                    y_true[-pos_len * self.count_negative_sample:]))
            else:
                y_real_true = y_true
            return F.binary_cross_entropy(y_pred, y_real_true).mean()

        def _run_train_step(engine, batch):
            self.model.train()
            optimizer.zero_grad()
            user_batch, pos_item_batch = batch
            neg_item_batch = self._get_neg_batch(user_batch)
            pos_relevance = self.model(user_batch.to(self.device),
                                       pos_item_batch.to(self.device))
            neg_relevance = self.model(
                user_batch.repeat([self.count_negative_sample])
                .to(self.device),
                neg_item_batch.to(self.device))
            y_pred = torch.cat((pos_relevance, neg_relevance), 0)
            loss = _loss(y_pred, y_true)
            loss.backward()
            optimizer.step()

            return loss.item()

        def _run_val_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                user_batch, pos_item_batch = batch
                neg_item_batch = self._get_neg_batch(user_batch)
                pos_relevance = self.model(user_batch.to(self.device),
                                           pos_item_batch.to(self.device))
                neg_relevance = self.model(user_batch.repeat([
                    self.count_negative_sample]).to(self.device),
                    neg_item_batch.to(self.device))
                y_pred = torch.cat((pos_relevance, neg_relevance), 0)
                return y_pred, y_true

        self.trainer = Engine(_run_train_step)
        self.val_evaluator = Engine(_run_val_step)

        Loss(_loss).attach(self.val_evaluator, 'loss')
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_loss(trainer):
            self.logger.debug("Epoch[{}] current loss: {:.4f}"
                              .format(trainer.state.epoch,
                                      trainer.state.output))

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.val_evaluator.run(val_data_loader)
            metrics = self.val_evaluator.state.metrics
            self.logger.debug("Epoch[{}] validation average loss: {:.4f}"
                              .format(trainer.state.epoch, metrics['loss']))

        def score_function(engine):
            return -engine.state.metrics['loss']

        early_stopping = EarlyStopping(patience=self.patience,
                                       score_function=score_function,
                                       trainer=self.trainer)
        self.val_evaluator.add_event_handler(Events.COMPLETED,
                                             early_stopping)
        checkpoint = ModelCheckpoint(
            self.spark.conf.get("spark.local.dir"),
            create_dir=True,
            require_empty=False,
            n_saved=self.n_saved,
            score_function=score_function,
            score_name="loss",
            filename_prefix="best",
            global_step_transform=global_step_from_engine(self.trainer))

        self.val_evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                             checkpoint, {"nmf": self.model})
        self.trainer.run(train_data_loader, max_epochs=self.epochs)

    @staticmethod
    def _predict_by_user(
            pandas_df: pd.DataFrame,
            model: nn.Module,
            items_np: np.array,
            k: int,
            items_count: int
    ) -> pd.DataFrame:
        user_idx = pandas_df["user_idx"][0]
        cnt = min(len(pandas_df) + k, len(items_np))

        model.eval()
        with torch.no_grad():
            user_batch = LongTensor([user_idx] * len(items_np))
            item_batch = LongTensor(items_np)
            user_recs = model(user_batch, item_batch).detach()
            best_item_idx = (torch.argsort(
                user_recs,
                descending=True)[:cnt]).numpy()
            return pd.DataFrame({
                "user_idx": cnt * [user_idx],
                "item_idx": items_np[best_item_idx],
                "relevance": user_recs[best_item_idx]
            })
