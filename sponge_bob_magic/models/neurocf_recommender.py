"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas
import torch
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from torch.nn import Embedding, Module, DataParallel
from torch.utils.data import DataLoader, TensorDataset
import torch.optim

from sponge_bob_magic import utils
from sponge_bob_magic.constants import DEFAULT_CONTEXT
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
    num_workers: int = 10

    def __init__(self, spark: SparkSession,
                 learning_rate: float = 0.5,
                 epochs: int = 1,
                 batch_size: int = 1000,
                 embedding_dimension: int = 10):
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

        self.user_indexer = StringIndexer(inputCol="user_id",
                                          outputCol="user_idx")
        self.item_indexer = StringIndexer(inputCol="item_id",
                                          outputCol="item_idx")

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
        self.user_indexer_model = self.user_indexer.fit(log)
        self.item_indexer_model = self.item_indexer.fit(log)

        log_indexed = self.user_indexer_model.transform(log)
        log_indexed = self.item_indexer_model.transform(log_indexed)

        self.num_users = log_indexed.select("user_idx").distinct().count()
        self.num_items = log_indexed.select("item_idx").distinct().count()

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

    def _run_single_batch(self, batch):
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

        return loss

    def _run_epoch(self,
                   user_batch: torch.LongTensor,
                   item_batch: torch.LongTensor,
                   optimizer: torch.optim.Optimizer):
        loss_value = 0

        data = DataLoader(
            TensorDataset(user_batch, item_batch),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        current_loss = None
        for batch in data:
            loss = self._run_single_batch(batch)

            loss_value += loss.item()
            current_loss = loss_value / len(data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return current_loss

    def _fit_partial(self, log: DataFrame, user_features: Optional[DataFrame],
                     item_features: Optional[DataFrame],
                     path: Optional[str] = None) -> None:
        logging.debug("Индексирование данных")
        log_indexed = self.user_indexer_model.transform(log)
        log_indexed = self.item_indexer_model.transform(log_indexed)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        logging.debug("Составление батча:")
        tensor_data = NeuroCFRecommender.spark2pandas_csv(
            log_indexed.select("user_idx", "item_idx"),
            os.path.join(path, "tmp_tensor_data")
        )

        user_batch = torch.LongTensor(tensor_data["user_idx"].values)
        item_batch = torch.LongTensor(tensor_data["item_idx"].values)

        logging.debug("Обучение модели")
        for epoch in range(self.epochs):
            logging.debug(f"-- Эпоха {epoch}")
            current_loss = self._run_epoch(user_batch, item_batch, optimizer)
            logging.debug(f"-- Текущее значение: {current_loss:.4f}")

    def get_loss(
            self,
            log: DataFrame,
            path: str
    ) -> float:
        """
        Считает значение функции потерь на одной эпохе.
        Записывает временные файлы.

        :param path: путь, по которому записывается временные файлы
        :param log: лог пользователей и объектов в стандартном формате
        :return: значение функции потерь
        """
        log_indexed = self.user_indexer_model.transform(log)
        log_indexed = self.user_indexer_model.transform(log_indexed)

        self.model.eval()

        loss_value = 0

        logging.debug("Составление батча:")
        tensor_data = NeuroCFRecommender.spark2pandas_csv(
            log_indexed.select("user_idx", "item_idx"),
            os.path.join(path, "tmp_tensor_data")
        )

        user_batch = torch.LongTensor(tensor_data["user_idx"].values)
        item_batch = torch.LongTensor(tensor_data["item_idx"].values)

        data = DataLoader(
            TensorDataset(user_batch, item_batch),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        with torch.no_grad():
            for batch in data:
                loss = self._run_single_batch(batch)
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
        logging.debug("Индексирование пользователей и объектов")
        users_indexed = NeuroCFRecommender.spark2pandas_csv(
            self.user_indexer_model.transform(users),
            os.path.join(path, "tmp_users_indexed")
        )

        items_indexed = NeuroCFRecommender.spark2pandas_csv(
            self.item_indexer_model.transform(items),
            os.path.join(path, "tmp_items_indexed")
        )

        self.model.eval()

        users = list(users_indexed["user_idx"].values)
        items = list(items_indexed["item_idx"].values)
        user_count = len(users)
        item_count = len(items)

        logging.debug("Создание тензоров")
        user_ids = torch.from_numpy(
            np.repeat(users, item_count).astype(int)).cpu()
        item_ids = torch.from_numpy(
            np.tile(items, user_count).astype(int)).cpu()

        if torch.cuda.is_available():
            user_ids = user_ids.cuda()
            item_ids = item_ids.cuda()

        logging.debug("Предсказание модели")
        with torch.no_grad():
            sparse_recs = (
                self.model.forward(user=user_ids, item=item_ids)
                    .reshape(user_count, item_count)
                    .detach()
                    .cpu()
                    .to_sparse()
            )

        logging.debug("Преобразование numpy-массива в спарк-датафрейм")
        # ToDo: придумать, как по-нормальному передать результат из спарка
        arr = np.c_[sparse_recs.indices().T, sparse_recs.values()]
        columns = ["user_idx", "item_idx", "relevance"]
        recs = NeuroCFRecommender.numpy2spark(
            self.spark, arr, columns,
            os.path.join(path, "tmp_recs_array.csv")
        )
        for column in columns[:2]:
            recs = recs.withColumn(column, sf.col(column).cast("int"))
        recs = recs.withColumn("relevance", sf.col("relevance").cast("double"))
        recs = recs.withColumn("context", sf.lit(DEFAULT_CONTEXT))

        logging.debug("Обратное преобразование индексов")
        user_converter = IndexToString(inputCol="user_idx",
                                       outputCol="user_id",
                                       labels=self.user_indexer_model.labels)
        item_converter = IndexToString(inputCol="item_idx",
                                       outputCol="item_id",
                                       labels=self.item_indexer_model.labels)

        recs = user_converter.transform(recs)
        recs = item_converter.transform(recs)
        recs = recs.drop("user_idx", "item_idx")

        if to_filter_seen_items:
            recs = self._filter_seen_recs(recs, log)

        recs = self._get_top_k_recs(recs, k)

        logging.debug("Преобразование отрицательных relevance")
        recs = NeuroCFRecommender.min_max_scale_column(recs, "relevance")

        if path is not None:
            logging.debug("Запись на диск рекомендаций")
            recs = utils.write_read_dataframe(
                self.spark, recs,
                os.path.join(path, "recs.parquet")
            )

        return recs.cache()

    @staticmethod
    def min_max_scale_column(df: DataFrame, column: str) -> DataFrame:
        """
        Отнормировать колонку датафрейма.
        Применяет классическую форму нормализации с минимумом и максимумом:
        new_value_i = (value_i - min) / (max - min).

        :param df: спарк-датафрейм
        :param column: имя колонки, которую надо нормализовать
        :return: исходный датафрейм с измененной колонкой
        """
        max_value = (
            df
            .agg({column: "max"})
            .collect()[0][0]
        )
        min_value = (
            df
            .agg({column: "min"})
            .collect()[0][0]
        )

        df = df.withColumn(
            column,
            (sf.col(column) - min_value) / (max_value - min_value)
        )
        return df

    @staticmethod
    def spark2pandas_csv(df: DataFrame, path: str) -> pandas.DataFrame:
        """
        Преобразовать спарк-датафрейм в пандас-датафрейм.
        Функция записывает спарк-датафрейм на диск в виде CSV,
        а затем pandas считывает этот файл в виде пандас-датафрейма.
        Создается временный файл по пути `path`.

        :param df: спарк-датафрейм, который надо переобразовать в пандас
        :param path: путь, по которому будет записан датафрейм и заново считан
        :return:
        """
        logging.debug("-- Запись")
        (df
         .coalesce(1)
         .write
         .mode("overwrite")
         .csv(path, header=True))

        logging.debug("-- Считывание")
        pandas_path = os.path.join(path,
                                   [file
                                    for file in os.listdir(path)
                                    if file.endswith(".csv")][0])

        df_pd = pandas.read_csv(pandas_path)
        return df_pd

    @staticmethod
    def numpy2spark(spark, arr, columns, path, **kwargs):
        """
        Преобразовать numpy-массив в спарк-датафрейм.
        Функция записывает на на диск массив в формате текстового файла CSV,
        а затем спарк считывает его в формат датафрейма.
        Создается временный файл по пути `path`.

        :param spark: инициализированная спарк-сессия
        :param arr: numpy-массив, который необходио преобразовать в датафрейм
        :param columns: список названий колонок (в качестве header)
        :param path: путь, по которому записывается временный файл
        :param kwargs: дополнительные агрументы в функцию считывания CSV,
            которая принимает на вход функция `spark.read.csv`
        :return:
        """
        logging.debug("-- Запись")
        delimiter = ","
        np.savetxt(path, arr, delimiter=delimiter, fmt="%.5f",
                   header=delimiter.join(columns), comments="")

        logging.debug("-- Считывание")
        df = spark.read.csv(path, sep=delimiter, header=True, **kwargs)
        return df
