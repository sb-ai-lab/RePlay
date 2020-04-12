"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Engine, Events
from ignite.handlers import (EarlyStopping, ModelCheckpoint,
                             global_step_from_engine)
from ignite.metrics import Loss, RunningAverage
from scipy.sparse import csr_matrix
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.types import DoubleType
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn.functional as F
import torch.optim
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from sponge_bob_magic.models import Recommender
from sponge_bob_magic.session_handler import State
from sponge_bob_magic.utils import get_top_k_recs


class VAE(nn.Module):
    """Простой вариационный авто кодировщик"""
    def __init__(
            self,
            item_count: int,
            latent_dim: int,
            p_dims: Optional[List[int]] = None,
            q_dims: Optional[List[int]] = None,
            dropout: float = 0.3
    ):
        """
        Инициализация модели. Создает эмбеддинги пользователей и объектов.

        :param user_count: количество пользователей
        :param item_count: количество объектов
        :param embedding_dimension: размерность представления пользователей и
            объектов
        """
        super().__init__()
        if p_dims:
            if q_dims:
                self.q_dims = q_dims
            else:
                self.q_dims = p_dims[::-1]
            self.p_dims = p_dims
        else:
            self.q_dims = []
            self.p_dims = []
        self.latent_dim = latent_dim
        self.q_dims = [item_count] + self.q_dims + [latent_dim * 2]
        self.p_dims = [latent_dim] + self.p_dims + [item_count]

        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in
                                       zip(self.q_dims[:-1], self.q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in
                                       zip(self.p_dims[:-1], self.p_dims[1:])])
        self.dropout = nn.Dropout(dropout)
        self.activation = torch.relu

        for layer in self.q_layers:
            self.weight_init(layer)

        for layer in self.p_layers:
            self.weight_init(layer)

    def encode(self, batch: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Энкодер"""
        hidden = F.normalize(batch, p=2, dim=1)
        hidden = self.dropout(hidden)

        for layer in self.q_layers[:-1]:
            hidden = layer(hidden)
            hidden = self.activation(hidden)

        hidden = self.q_layers[-1](hidden)
        mu_latent = hidden[:, :self.latent_dim]
        logvar_latent = hidden[:, self.latent_dim:]
        return mu_latent, logvar_latent

    def reparameterize(self, mu_latent: torch.Tensor,
                       logvar_latent: torch.Tensor) -> torch.Tensor:
        """Репараметризационный трюк, необходимый для обрабного прохождения
        сигнала по семплированным данным"""

        if self.training:
            std = torch.exp(0.5 * logvar_latent)
            eps = torch.randn_like(std)
            return eps * std + mu_latent
        return mu_latent

    def decode(self, z_latent: torch.Tensor) -> torch.Tensor:
        """Декодер"""
        hidden = z_latent
        for layer in self.p_layers[:-1]:
            hidden = layer(hidden)
            hidden = self.activation(hidden)
        return self.p_layers[-1](hidden)

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor]:
        """
        Один проход нейросети.

        :param batch: батч пользователей
        :return: батч реконструированнх пользователей, а также матожидание и
        логарифм отклонения латентного распределения
        """
        mu_latent, logvar_latent = self.encode(batch)
        z_latent = self.reparameterize(mu_latent, logvar_latent)
        return self.decode(z_latent), mu_latent, logvar_latent

    @staticmethod
    def weight_init(layer: nn.Module):
        """
        Инициализация весов линейного слоя методом Хавьера

        :param layer: слой нейронной сети
        """
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight.data)
            layer.bias.data.normal_(0.0, 0.001)


class VAERec(Recommender):
    """
    Вариационный автокодировщик. Общая схема его работы
    представлена на рисунке.

    .. image:: /images/vae-gaussian.png

    """
    num_workers: int = 0
    batch_size_users: int = 5000
    patience: int = 5
    n_saved: int = 2

    def __init__(self, learning_rate: float = 0.05,
                 epochs: int = 1,
                 latent_dim: int = 10,
                 p_dims: Optional[List[int]] = None,
                 q_dims: Optional[List[int]] = None,
                 dropout: float = 0.3,
                 anneal: float = 0.005,
                 l2: float = 0):
        """
        Инициализирует параметры модели и сохраняет спарк-сессию.

        :param learning_rate: шаг обучения
        :param epochs: количество эпох, в течение которых учимся
        :param latent_dim: размер скрытого представления пользователя
        :param p_dims: последовательность размеров слоев декодера
        :param q_dims: последовательность размеров слоев енкодера
        :param dropout: коэффициент дропаута
        :param anneal: коэффициент отжига от 0 до 1
        :param l2: коэффициент l2 регуляризации
        """
        self.device = State().device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.p_dims = p_dims
        self.q_dims = q_dims
        self.dropout = dropout
        self.anneal = anneal
        self.l2 = l2

    def get_params(self) -> Dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "latent_dim": self.latent_dim
        }

    def _get_data_loader(self,
                         data: pd.DataFrame,
                         shuffle: bool = True
                         ) -> (csr_matrix, DataLoader, np.array):
        """Функция получения загрузчика данных, а также матрицы с данными"""
        users_count = data["user_idx"].value_counts().count()
        user_idx = data.user_idx.astype('category').cat
        user_batch = csr_matrix((np.ones(len(data.user_idx)),
                                 ([user_idx.codes.values,
                                   data.item_idx.values])),
                                shape=(users_count, self.items_count))
        data_loader = DataLoader(
            TensorDataset(torch.arange(users_count).long()),
            batch_size=self.batch_size_users,
            shuffle=shuffle,
            num_workers=self.num_workers)

        return user_batch, data_loader, user_idx.categories.values

    def _fit(self,
             log: DataFrame,
             user_features: Optional[DataFrame] = None,
             item_features: Optional[DataFrame] = None) -> None:
        self.logger.debug("Индексирование данных")
        log_indexed = self.user_indexer.transform(log)
        log_indexed = self.item_indexer.transform(log_indexed)

        self.logger.debug("Составление батча:")
        data = log_indexed.select("user_idx", "item_idx").toPandas()
        splitter = GroupShuffleSplit(n_splits=1,
                                     test_size=0.1,
                                     random_state=42)
        train_idx, valid_idx = next(splitter.split(data, groups=data[
            "user_idx"]))
        train_data, valid_data = data.iloc[train_idx], data.iloc[valid_idx]

        train_user_batch, train_data_loader, _ = self._get_data_loader(
            train_data)
        valid_user_batch, valid_data_loader, _ = self._get_data_loader(
            valid_data, False)

        self.logger.debug("Обучение модели")
        self.model = VAE(
            item_count=self.items_count,
            latent_dim=self.latent_dim,
            p_dims=self.p_dims,
            q_dims=self.q_dims,
            dropout=self.dropout).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2 / self.batch_size_users)
        lr_scheduler = ExponentialLR(optimizer, gamma=0.98)
        scheduler = LRScheduler(lr_scheduler)

        def _loss(x_pred, x_true, mu_latent,
                  logvar_latent, anneal=self.anneal):
            log_softmax_var = F.log_softmax(x_pred, dim=1)
            bce = - (log_softmax_var * x_true).sum(dim=1).mean()
            kld = -0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) -
                                   logvar_latent.exp(), dim=1).mean()
            return bce + anneal * kld

        def _run_train_step(engine, batch):
            self.model.train()
            optimizer.zero_grad()
            user_batch = torch.FloatTensor(train_user_batch[batch[0]]
                                           .toarray()).to(self.device)
            pred_user_batch, latent_mu, latent_logvar = self.model(user_batch)
            loss = _loss(pred_user_batch, user_batch, latent_mu, latent_logvar)
            loss.backward()
            optimizer.step()
            return loss.item()

        def _run_val_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                user_batch = torch.FloatTensor(valid_user_batch[batch[0]]
                                               .toarray()).to(self.device)
                pred_user_batch, latent_mu, latent_logvar = self.model(
                    user_batch)
                return (pred_user_batch,
                        user_batch,
                        {"mu_latent": latent_mu,
                         "logvar_latent": latent_logvar})

        vae_trainer = Engine(_run_train_step)
        val_evaluator = Engine(_run_val_step)

        Loss(_loss).attach(val_evaluator, 'elbo')
        vae_trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)
        avg_output = RunningAverage(output_transform=lambda x: x)
        avg_output.attach(vae_trainer, 'avg')

        @vae_trainer.on(Events.EPOCH_COMPLETED)
        def log_training_loss(trainer):
            self.logger.debug("Epoch[{}] current elbo: {:.5f}"
                              .format(trainer.state.epoch,
                                      trainer.state.metrics["avg"]))

        @vae_trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            val_evaluator.run(valid_data_loader)
            metrics = val_evaluator.state.metrics
            self.logger.debug("Epoch[{}] validation average elbo: {:.5f}"
                              .format(trainer.state.epoch, metrics['elbo']))

        def score_function(engine):
            return -engine.state.metrics['elbo']

        early_stopping = EarlyStopping(patience=self.patience,
                                       score_function=score_function,
                                       trainer=vae_trainer)
        val_evaluator.add_event_handler(Events.COMPLETED,
                                        early_stopping)
        checkpoint = ModelCheckpoint(
            self.spark.conf.get("spark.local.dir"),
            create_dir=True,
            require_empty=False,
            n_saved=self.n_saved,
            score_function=score_function,
            score_name="elbo",
            filename_prefix="best",
            global_step_transform=global_step_from_engine(vae_trainer))

        val_evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                        checkpoint, {"vae": self.model})
        vae_trainer.run(train_data_loader, max_epochs=self.epochs)

    def _predict(self,
                 log: DataFrame,
                 k: int,
                 users: Optional[DataFrame] = None,
                 items: Optional[DataFrame] = None,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        self.logger.debug("Индексирование данных")
        log_indexed = self.user_indexer.transform(
            users.join(log, how="left", on="user_id")
            .select("user_id", "item_id"))

        log_indexed = self.item_indexer.transform(log_indexed)

        self.logger.debug("Предсказание модели")
        log_data = log_indexed.select("user_idx", "item_idx").toPandas()
        log_user_batch, log_data_loader, user_idx = self._get_data_loader(
            log_data, False)

        predictions = pd.DataFrame(
            columns=["user_idx", "item_idx", "relevance"])

        def _run_pred_step(engine, batch):
            nonlocal predictions
            self.model.eval()
            with torch.no_grad():
                user_batch = torch.FloatTensor(log_user_batch[batch[0]]
                                               .toarray())
                cnt_by_user = (user_batch > 0) .sum(dim=1)
                user_batch_idx = user_idx[batch[0]]
                pred_user_batch, _, _ = self.model(user_batch.to(self.device))
                for user_id, user_rec, cnt in zip(
                        user_batch_idx,
                        pred_user_batch.detach().cpu(),
                        cnt_by_user):
                    best_item_idx = torch.argsort(user_rec,
                                                  descending=True)[:cnt+k]
                    predictions = predictions.append(
                        pd.DataFrame({"user_idx": [user_id] * (cnt + k),
                                          "item_idx": best_item_idx,
                                          "relevance": user_rec[best_item_idx]
                                          }), sort=False)

        pred_evaluator = Engine(_run_pred_step)
        pred_evaluator.run(log_data_loader)
        recs = self.spark.createDataFrame(predictions)

        self.logger.debug("Обратное преобразование индексов")
        recs = self.inv_item_indexer.transform(recs)
        recs = self.inv_user_indexer.transform(recs)
        recs = recs.drop("user_idx", "item_idx")

        if filter_seen_items:
            recs = self._filter_seen_recs(recs, log)

        recs = get_top_k_recs(recs, k)
        self.logger.debug("Преобразование отрицательных relevance")
        recs = VAERec.min_max_scale_column(recs, "relevance")

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
        unlist = sf.udf(lambda x: float(list(x)[0]), DoubleType())
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

    def load_model(self, path: str) -> None:
        """
        Загрузка весов модели из файла

        :param path: путь к файлу, откуда загружать
        :return:
        """
        self.logger.debug("-- Загрузка модели из файла")
        self.model.load_state_dict(torch.load(path))
