"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
import shutil
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas
import torch.optim
from annoy import AnnoyIndex
from pyspark.ml import Pipeline
from pyspark.ml.feature import (IndexToString, MinMaxScaler, StringIndexer,
                                StringIndexerModel, VectorAssembler)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from torch import Tensor
from torch.nn import DataParallel, Embedding, Module
from torch.utils.data import DataLoader, TensorDataset

from sponge_bob_magic.constants import DEFAULT_CONTEXT
from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import get_top_k_recs


class NMF(Module):
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
        user_biases = Embedding(num_embeddings=user_count,
                                embedding_dim=1)

        user_embedding.weight.data.normal_(0, 1.0 / embedding_dimension)
        item_embedding.weight.data.normal_(0, 1.0 / embedding_dimension)
        user_biases.weight.data.zero_()
        item_biases.weight.data.zero_()

        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.item_biases = item_biases
        self.user_biases = user_biases

    def forward(self, user: torch.Tensor, item: torch.Tensor, get_embs=False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Один проход нейросети.

        :param user: батч ID пользователей
        :param item: батч ID объектов
        :type get_embs: флаг, указывающий, возвращать ли промежуточные эмбеддинги
        :return: батч весов предсказанных взаимодействий пользователей и
            объектов или батч промежуточных эмбеддингов
        """
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        dot = (user_emb * item_emb).sum(dim=1).squeeze()
        relevance = (
            dot + self.item_biases(item).squeeze() +
            self.user_biases(user).squeeze()
        )
        if get_embs:
            return user_emb, item_emb
        else:
            return relevance


class NeuroMFRec(Recommender):
    """ Модель матричной факторизации на нейросети. """
    num_workers: int = 10
    batch_size_fit_users: int = 100000
    batch_size_predict_users: int = 100
    batch_size_predict_items: int = 10000
    user_indexer_model: StringIndexerModel
    item_indexer_model: StringIndexerModel
    num_users: int
    num_items: int
    num_trees_annoy: int = 10

    def __init__(self, learning_rate: float = 0.5,
                 epochs: int = 1,
                 embedding_dimension: int = 10):
        """
        Инициализирует параметры модели и сохраняет спарк-сессию.

        :param learning_rate: шаг обучения
        :param epochs: количество эпох, в течение которых учимся
        :param embedding_dimension: размер представления пользователей/объектов
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.embedding_dimension = embedding_dimension

        self.user_indexer = StringIndexer(inputCol="user_id",
                                          outputCol="user_idx")
        self.item_indexer = StringIndexer(inputCol="item_id",
                                          outputCol="item_idx")

        self.annoy_index = AnnoyIndex(embedding_dimension, "angular")

    def get_params(self) -> Dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "embedding_dimension": self.embedding_dimension
        }

    def _pre_fit(self, log: DataFrame,
                 user_features: Optional[DataFrame],
                 item_features: Optional[DataFrame]) -> None:
        self.user_indexer_model = self.user_indexer.fit(log)
        self.item_indexer_model = self.item_indexer.fit(log)

        log_indexed = self.user_indexer_model.transform(log)
        log_indexed = self.item_indexer_model.transform(log_indexed)

        self.num_users = log_indexed.select("user_idx").distinct().count()
        self.num_items = log_indexed.select("item_idx").distinct().count()

        model = NMF(
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

    def _run_single_batch(self, batch: Tensor) -> Tensor:
        negative_items = torch.from_numpy(
            np.random.randint(low=0, high=self.num_items - 1,
                              size=batch[0].shape[0])
        )

        if torch.cuda.is_available():
            batch = [item.cuda() for item in batch]
            negative_items = negative_items.cuda()

        positive_relevance = self.model.forward(batch[0], batch[1])
        negative_relevance = self.model.forward(batch[0], negative_items)

        loss = torch.clamp(
            negative_relevance - positive_relevance + 1.0, 0.0, 1.0
        ).mean()

        return loss

    def _run_epoch(self,
                   user_batch: torch.LongTensor,
                   item_batch: torch.LongTensor,
                   optimizer: torch.optim.Optimizer) -> float:
        loss_value = 0.0
        data = DataLoader(
            TensorDataset(user_batch, item_batch),
            batch_size=self.batch_size_fit_users,
            shuffle=True,
            num_workers=self.num_workers
        )
        current_loss = 0.0
        for batch in data:
            loss = self._run_single_batch(batch)
            loss_value += loss.item()
            current_loss = loss_value / len(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return current_loss

    def _fit_partial(self, log: DataFrame, user_features: Optional[DataFrame],
                     item_features: Optional[DataFrame]) -> None:
        logging.debug("Индексирование данных")
        log_indexed = self.user_indexer_model.transform(log)
        log_indexed = self.item_indexer_model.transform(log_indexed)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        logging.debug("Составление батча:")
        spark = SparkSession(log.rdd.context)
        tensor_data = NeuroMFRec.spark2pandas_csv(
            log_indexed.select("user_idx", "item_idx"),
            os.path.join(spark.conf.get("spark.local.dir"),
                         "tmp_tensor_data")
        )
        user_batch = torch.LongTensor(tensor_data["user_idx"].values)
        item_batch = torch.LongTensor(tensor_data["item_idx"].values)
        logging.debug("Обучение модели")
        for epoch in range(self.epochs):
            logging.debug("-- Эпоха %d", epoch)
            current_loss = self._run_epoch(user_batch, item_batch, optimizer)
            logging.debug("-- Текущее значение: %.4f", current_loss)

        self.model.eval()
        logging.debug("-- Запись annoy индексов")
        if torch.cuda.is_available():
            user_batch = user_batch.cuda()
            item_batch = item_batch.cuda()
        _, item_embs = self.model(user_batch, item_batch, get_embs=True)
        for item_id, item_emb in zip(
                tensor_data["item_idx"].values,
                item_embs.detach().cpu().numpy()
        ):
            self.annoy_index.add_item(int(item_id), item_emb)
        self.annoy_index.build(self.num_trees_annoy)

    def _predict(self,
                 k: int,
                 users: DataFrame, items: DataFrame,
                 context: str, log: DataFrame,
                 user_features: Optional[DataFrame],
                 item_features: Optional[DataFrame],
                 to_filter_seen_items: bool = True) -> DataFrame:
        self.model.eval()
        sep = ","
        spark = SparkSession(users.rdd.context)
        tmp_path = os.path.join(spark.conf.get("spark.local.dir"), "recs")
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        os.makedirs(tmp_path)

        logging.debug("Индексирование данных")
        users = self.user_indexer_model.transform(users)

        logging.debug("Предсказание модели")
        tensor_data = NeuroMFRec.spark2pandas_csv(
            users.select("user_idx"),
            os.path.join(spark.conf.get("spark.local.dir"),
                         "tmp_tensor_data")
        )
        user_batch = torch.LongTensor(tensor_data["user_idx"].values)
        item_batch = torch.ones_like(user_batch)
        if torch.cuda.is_available():
            user_batch = user_batch.cuda()
            item_batch = item_batch.cuda()
        user_embs, _ = self.model(user_batch, item_batch, get_embs=True)
        predictions = pandas.DataFrame(columns=["user_idx", "item_idx"])
        logging.debug("Поиск ближайших айтемов с помощью annoy")
        for user_id, user_emb in zip(tensor_data["user_idx"].values,
                                     user_embs.detach().cpu().numpy()):
            pred_for_user, relevance = self.annoy_index.get_nns_by_vector(
                user_emb, k, include_distances=True
            )
            predictions = predictions.append(
                pandas.DataFrame({"user_idx": [user_id] * k,
                                  "item_idx": pred_for_user,
                                  "relevance": relevance
                                  }), sort=False
            )
        predictions.to_csv(os.path.join(tmp_path, "predict.csv"),
                           sep=sep, header=True, index=False)

        recs = spark.read.csv(os.path.join(tmp_path, "predict.csv"),
                              sep=sep,
                              header=True,
                              inferSchema=True)
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

        recs = get_top_k_recs(recs, k)

        logging.debug("Преобразование отрицательных relevance")
        recs = NeuroMFRec.min_max_scale_column(recs, "relevance")

        return recs

    @staticmethod
    def min_max_scale_column(dataframe: DataFrame, column: str) -> DataFrame:
        """
        Отнормировать колонку датафрейма.
        Применяет классическую форму нормализации с минимумом и максимумом:
        new_value_i = (value_i - min) / (max - min).

        :param dataframe: спарк-датафрейм
        :param column: имя колонки, которую надо нормализовать
        :return: исходный датафрейм с измененной колонкой
        """
        unlist = udf(lambda x: float(list(x)[0]), DoubleType())
        assembler = VectorAssembler(
            inputCols=[column], outputCol=column+"_Vect"
        )
        scaler = MinMaxScaler(
            inputCol=column+"_Vect", outputCol=column+"_Scaled"
        )
        pipeline = Pipeline(stages=[assembler, scaler])
        dataframe = (pipeline
                     .fit(dataframe)
                     .transform(dataframe)
                     .withColumn(column, unlist(column+"_Scaled"))
                     .drop(column+"_Vect", column+"_Scaled"))

        return dataframe

    @staticmethod
    def spark2pandas_csv(dataframe: DataFrame, path: str) -> pandas.DataFrame:
        """
        Преобразовать спарк-датафрейм в пандас-датафрейм.
        Функция записывает спарк-датафрейм на диск в виде CSV,
        а затем pandas считывает этот файл в виде пандас-датафрейма.
        Создается временный файл по пути `path`.

        :param dataframe: спарк-датафрейм, который надо переобразовать в пандас
        :param path: путь, по которому будет записан датафрейм и заново считан
        :return:
        """
        logging.debug("-- Запись")
        (dataframe
         .coalesce(1)
         .write
         .mode("overwrite")
         .csv(path, header=True))

        logging.debug("-- Считывание")
        pandas_path = os.path.join(path,
                                   [file
                                    for file in os.listdir(path)
                                    if file.endswith(".csv")][0])
        pandas_dataframe = pandas.read_csv(pandas_path)
        return pandas_dataframe
