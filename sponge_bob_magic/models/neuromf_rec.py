"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
import shutil
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas
from annoy import AnnoyIndex
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Engine, Events
from ignite.handlers import (EarlyStopping, ModelCheckpoint,
                             global_step_from_engine)
from ignite.metrics import Loss
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from sklearn.model_selection import train_test_split
import torch.optim
from torch import LongTensor, Tensor
from torch.nn import Embedding, Module
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from sponge_bob_magic.models import Recommender
from sponge_bob_magic.session_handler import State
from sponge_bob_magic.utils import get_top_k_recs


class NMF(Module):
    "Простая нейронная сеть, соответствующая колоборативной фильтрации"
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
        :type get_embs: флаг, указывающий, возвращать ли промежуточные
            эмбеддинги
        :return: батч весов предсказанных взаимодействий пользователей и
            объектов или батч промежуточных эмбеддингов
        """
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        dot = (user_emb * item_emb).sum(dim=1).squeeze()
        relevance = (dot + self.item_biases(item).squeeze() +
                     self.user_biases(user).squeeze())
        if get_embs:
            return user_emb, item_emb
        return relevance


class NeuroMFRec(Recommender):
    """
    Эта модель является вариацей на модель из статьи Neural Matrix Factorization
    (NeuMF, NCF)
    """
    num_workers: int = 10
    batch_size_fit_users: int = 100000
    batch_size_predict_users: int = 100
    batch_size_predict_items: int = 10000
    num_users: int
    num_items: int
    num_trees_annoy: int = 10
    trainer: Engine
    val_evaluator: Engine
    train_evaluator: Engine
    patience: int = 3
    n_saved: int = 1

    def __init__(self, learning_rate: float = 0.05,
                 epochs: int = 1,
                 embedding_dimension: int = 10):
        """
        Инициализирует параметры модели и сохраняет спарк-сессию.

        :param learning_rate: шаг обучения
        :param epochs: количество эпох, в течение которых учимся
        :param embedding_dimension: размер представления пользователей/объектов
        """
        self.device = State().device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.embedding_dimension = embedding_dimension
        self.count_negative_sample = 1
        self.annoy_index = AnnoyIndex(embedding_dimension, "angular")
        self.spark = State().session

    def get_params(self) -> Dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "embedding_dimension": self.embedding_dimension
        }

    def _pre_fit(self, log: DataFrame,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None) -> None:
        super()._pre_fit(log, user_features, item_features)
        log_indexed = self.user_indexer.transform(log)
        log_indexed = self.item_indexer.transform(log_indexed)
        self.num_users = log_indexed.select("user_idx").distinct().count()
        self.num_items = log_indexed.select("item_idx").distinct().count()
        self.model = NMF(
            user_count=self.num_users,
            item_count=self.num_items,
            embedding_dimension=self.embedding_dimension
        ).to(self.device)

    def _data_loader(self,
                     user_batch: Tensor,
                     item_batch: Tensor):
        dataset = TensorDataset(user_batch, item_batch)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size_fit_users,
            shuffle=True,
            num_workers=self.num_workers
        )
        return loader

    def _get_neg_batch(self, batch):
        negative_items = torch.from_numpy(
            np.random.randint(low=0, high=self.num_items - 1,
                              size=batch.shape[0] *
                              self.count_negative_sample)
        ).to(self.device)
        return negative_items

    def _fit(self,
             log: DataFrame,
             user_features: Optional[DataFrame] = None,
             item_features: Optional[DataFrame] = None) -> None:
        logging.debug("Индексирование данных")
        log_indexed = self.user_indexer.transform(log)
        log_indexed = self.item_indexer.transform(log_indexed)

        logging.debug("Составление батча:")
        tensor_data = NeuroMFRec.spark2pandas_csv(
            log_indexed.select("user_idx", "item_idx"),
            os.path.join(self.spark.conf.get("spark.local.dir"),
                         "tmp_tensor_data")
        )
        train_tensor_data, valid_tensor_data = train_test_split(
            tensor_data, stratify=tensor_data["user_idx"], test_size=0.1,
            random_state=42)
        train_user_batch = LongTensor(
            train_tensor_data["user_idx"].values
        ).to(self.device)
        train_item_batch = LongTensor(
            train_tensor_data["item_idx"].values
        ).to(self.device)
        valid_user_batch = LongTensor(
            valid_tensor_data["user_idx"].values
        ).to(self.device)
        valid_item_batch = LongTensor(
            valid_tensor_data["item_idx"].values
        ).to(self.device)
        logging.debug("Обучение модели")

        train_data_loader = DataLoader(
            TensorDataset(train_user_batch, train_item_batch),
            batch_size=self.batch_size_fit_users,
            shuffle=True,
            num_workers=self.num_workers
        )
        val_data_loader = DataLoader(
            TensorDataset(valid_user_batch, valid_item_batch),
            batch_size=self.batch_size_fit_users,
            shuffle=False,
            num_workers=self.num_workers
        )

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)
        lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
        scheduler = LRScheduler(lr_scheduler)
        y_true = torch.ones(self.batch_size_fit_users *
                            (1 + self.count_negative_sample)).to(self.device)
        y_true[self.batch_size_fit_users:] = -1

        def _loss(y_pred, y_true):
            pos_len = y_pred.shape[0] // (self.count_negative_sample + 1)
            negative_relevance = y_pred[pos_len:]
            positive_relevance = y_pred[:pos_len]
            return torch.clamp(
                negative_relevance - positive_relevance + 1.0, 0.0, 1.0
            ).mean()

        def _run_train_step(engine, batch):
            self.model.train()
            optimizer.zero_grad()
            user_batch, pos_item_batch = batch
            neg_item_batch = self._get_neg_batch(user_batch)
            pos_relevance = self.model(user_batch, pos_item_batch)
            neg_relevance = self.model(user_batch, neg_item_batch)
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
                pos_relevance = self.model(user_batch, pos_item_batch)
                neg_relevance = self.model(user_batch, neg_item_batch)
                y_pred = torch.cat((pos_relevance, neg_relevance), 0)
                return y_pred, y_true

        self.trainer = Engine(_run_train_step)
        self.train_evaluator = Engine(_run_val_step)
        self.val_evaluator = Engine(_run_val_step)

        Loss(_loss).attach(self.train_evaluator, 'loss')
        Loss(_loss).attach(self.val_evaluator, 'loss')
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_loss(trainer):
            print("Epoch[{}] Loss: {:.4f}".format(
                trainer.state.epoch, trainer.state.output))

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            self.train_evaluator.run(train_data_loader)
            metrics = self.train_evaluator.state.metrics
            print("Training set Results - Epoch: {} Avg loss: {:.4f}"
                  .format(trainer.state.epoch, metrics['loss']))

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.val_evaluator.run(val_data_loader)
            metrics = self.val_evaluator.state.metrics
            print("Validation set Results - Epoch: {} Avg loss: {:.4f}"
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
            filename_prefix='best',
            global_step_transform=global_step_from_engine(
                self.trainer))

        self.trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                       checkpoint, {'nmf': self.model})
        self.trainer.run(train_data_loader, max_epochs=self.epochs)

        self.model.eval()
        logging.debug("-- Запись annoy индексов")
        _, item_embs = self.model(train_user_batch,
                                  train_item_batch,
                                  get_embs=True)
        for item_id, item_emb in zip(
                train_tensor_data["item_idx"].values,
                item_embs.detach().cpu().numpy()
        ):
            self.annoy_index.add_item(int(item_id), item_emb)
        self.annoy_index.build(self.num_trees_annoy)

    def _predict(self,
                 log: DataFrame,
                 k: int,
                 users: Optional[DataFrame] = None,
                 items: Optional[DataFrame] = None,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        self.model.eval()
        sep = ","
        tmp_path = os.path.join(self.spark.conf.get("spark.local.dir"), "recs")
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        os.makedirs(tmp_path)

        logging.debug("Индексирование данных")
        users = self.user_indexer.transform(users)

        logging.debug("Предсказание модели")
        tensor_data = NeuroMFRec.spark2pandas_csv(
            users.select("user_idx"),
            os.path.join(self.spark.conf.get("spark.local.dir"),
                         "tmp_tensor_users_data")
        )
        user_batch = LongTensor(tensor_data["user_idx"].values).to(self.device)
        item_batch = torch.ones_like(user_batch).to(self.device)
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

        recs = self.spark.read.csv(os.path.join(tmp_path, "predict.csv"),
                              sep=sep,
                              header=True,
                              inferSchema=True)

        logging.debug("Обратное преобразование индексов")
        recs = self.inv_item_indexer.transform(recs)
        recs = self.inv_user_indexer.transform(recs)
        recs = recs.drop("user_idx", "item_idx")

        if filter_seen_items:
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
            inputCols=[column], outputCol=column + "_Vect"
        )
        scaler = MinMaxScaler(
            inputCol=column + "_Vect", outputCol=column + "_Scaled"
        )
        pipeline = Pipeline(stages=[assembler, scaler])
        dataframe = (pipeline
                     .fit(dataframe)
                     .transform(dataframe)
                     .withColumn(column, unlist(column + "_Scaled"))
                     .drop(column + "_Vect", column + "_Scaled"))

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

    def load_model(self, path: str) -> None:
        """
        Загрузка весов модели из файла

        :param path: путь к файлу, откуда загружать
        :return:
        """
        logging.debug("-- Загрузка модели из файла")
        self.model.load_state_dict(torch.load(path))
