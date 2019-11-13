"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from typing import Dict, Optional

import numpy as np
import torch
from pyspark.sql import DataFrame, SparkSession
from torch.nn import Embedding, Module, DataParallel
from torch.utils.data import DataLoader, TensorDataset

from sponge_bob_magic.models.base_recommender import BaseRecommender


class RecommenderModel(Module):
    def __init__(
            self,
            user_count: int,
            item_count: int,
            embedding_dimension: int
    ):
        """
        Инициализация модели. Создает эмбеддинги пользователей и объектов.

        :param user_count: количество пользователей
        :param item_count: количество объектов
        :param embedding_dimension: размерность представления пользователей и
            объектов
        """
        super().__init__()
        user_embedding = Embedding(num_embeddings=user_count,
                                   embedding_dim=embedding_dimension)
        item_embedding = Embedding(num_embeddings=item_count,
                                   embedding_dim=embedding_dimension)
        item_biases = Embedding(num_embeddings=item_count,
                                embedding_dim=1)
        user_biases = Embedding(num_embeddings=item_count,
                                embedding_dim=1)

        user_embedding.weight.data.normal_(0, 1.0 / embedding_dimension)
        item_embedding.weight.data.normal_(0, 1.0 / embedding_dimension)
        user_biases.weight.data.zero_()
        item_biases.weight.data.zero_()

        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.item_biases = item_biases
        self.user_biases = user_biases

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """
        Один проход нейросети.

        :param user: батч ID пользователей
        :param item: батч ID объектов
        :return: батч весов предсказанных взаимодействий пользователей и
            объектов
        """
        return (
                (self.user_embedding(user) * self.item_embedding(item))
                .sum(dim=1).squeeze() +
                self.item_biases(item).squeeze() +
                self.user_biases(user).squeeze()
        )


class NeuroCFRecommender(BaseRecommender):
    """ Модель на нейросети. """

    def __init__(self, spark: SparkSession,
                 learning_rate: float,
                 epochs: int,
                 batch_size: int,
                 embedding_dimension: int):
        """

        Инициализирует параметры модели и сохраняет спарк-сессию.

        :param embedding_dimension: размер представления пользователей/объектов
        :param spark: инициализированная спарк-сессия
        :param learning_rate: шаг обучения
        :param epochs: количество эпох, в течение которых учимся
        :param batch_size: размер батча для обучения
        """
        super().__init__(spark)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dimension = embedding_dimension

    def get_params(self) -> Dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "embedding_dimension": self.embedding_dimension
        }

    def _pre_fit(self, log: DataFrame,
                 user_features: Optional[DataFrame],
                 item_features: Optional[DataFrame],
                 path: Optional[str] = None) -> None:
        self.num_users = log.select("user_id").distinct().count()
        self.num_items = log.select("item_id").distinct().count()

        model = RecommenderModel(
            user_count=self.num_users,
            item_count=self.num_items,
            embedding_dimension=self.embedding_dimension
        )

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model = DataParallel(
                    model,
                    device_ids=list(range(torch.cuda.device_count()))
                ).cuda()
            else:
                self.model = model.cuda()
        else:
            self.model = model

    def _fit_partial(self, log: DataFrame, user_features: Optional[DataFrame],
                     item_features: Optional[DataFrame],
                     path: Optional[str] = None) -> None:
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        tensor_data = log.select("user_id", "item_id").toPandas()
        user_batch = torch.LongTensor(tensor_data["user_id"].values)
        item_batch = torch.LongTensor(tensor_data["item_id"].values)

        for epoch in range(self.epochs):
            loss_value = 0

            data = DataLoader(
                TensorDataset(user_batch, item_batch),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=20
            )

            for batch in data:
                negative_items = torch.from_numpy(
                    np.random.randint(low=0, high=self.num_items - 1,
                                      size=batch[0].shape[0])
                )

                if torch.cuda.is_available():
                    batch = [item.cuda() for item in batch]
                    negative_items = negative_items.cuda()

                positive_relevance = self.model.forward(batch[0], batch[1])
                negative_relevance = self.model.forward(
                    batch[0], negative_items
                )

                loss = torch.clamp(
                    negative_relevance - positive_relevance + 1.0, 0.0, 1.0
                ).mean()

                loss_value += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_loss = loss_value / len(data)
                logging.debug(f"Текущее значение: {current_loss:.4f}")

    def get_loss(
            self,
            log: DataFrame,
    ) -> float:
        """
        Считает значение функции потерь на одной эпохе.

        :param log: набор положительных примеров для обучения
        :return:
        """
        self.model.eval()

        loss_value = 0
        tensor_data = log.select("user_id", "item_id").toPandas()
        user_batch = torch.LongTensor(tensor_data.user_id.values)
        item_batch = torch.LongTensor(tensor_data.item_id.values)

        data = DataLoader(
            TensorDataset(user_batch, item_batch),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=20
        )
        with torch.no_grad():
            for batch in data:
                negative_items = torch.from_numpy(
                    np.random.randint(low=0, high=self.num_items - 1,
                                      size=batch[0].shape[0])
                )

                if torch.cuda.is_available():
                    batch = [item.cuda() for item in batch]
                    negative_items = negative_items.cuda()

                positive_relevance = self.model.forward(batch[0], batch[1])
                negative_relevance = self.model.forward(
                    batch[0], negative_items
                )

                loss = torch.clamp(
                    negative_relevance - positive_relevance + 1.0, 0.0, 1.0
                ).mean()

                loss_value += loss.item()
        return loss_value / len(data)

    def _predict(self,
                 k: int,
                 users: DataFrame, items: DataFrame,
                 context: str, log: DataFrame,
                 user_features: Optional[DataFrame],
                 item_features: Optional[DataFrame],
                 to_filter_seen_items: bool = True,
                 path: Optional[str] = None) -> DataFrame:
        self.model.eval()

        users = list(users.toPandas()["user_id"].values)
        items = list(items.toPandas()["item_id"].values)
        user_count = len(users)
        item_count = len(items)

        user_ids = torch.from_numpy(np.repeat(users, item_count))
        item_ids = torch.from_numpy(np.tile(items, user_count))

        with torch.no_grad():
            return (
                self.model.forward(user=user_ids, item=item_ids)
                    .reshape(user_count, item_count)
                    .detach().cpu().numpy()
            )
