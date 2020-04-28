"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pyspark.sql import DataFrame
from scipy.sparse import csr_matrix
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from sponge_bob_magic.models import TorchRecommender
from sponge_bob_magic.session_handler import State


class VAE(nn.Module):
    """Простой вариационный автокодировщик"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        item_count: int,
        latent_dim: int,
        decoder_dims: Optional[List[int]] = None,
        encoder_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        """
        Инициализация модели.

        :param item_count: количество объектов
        :param latent_dim: размерность скрытого представления
        :param decoder_dims: последовательность размеров скрытых слоев декодера
        :param encoder_dims: последовательность размеров скрытых слоев энкодера
        :param dropout: коэффициент дропаута
        """
        super().__init__()
        if decoder_dims:
            if encoder_dims:
                self.encoder_dims = encoder_dims
            else:
                self.encoder_dims = decoder_dims[::-1]
            self.decoder_dims = decoder_dims
        else:
            self.encoder_dims = []
            self.decoder_dims = []
        self.latent_dim = latent_dim
        self.encoder_dims = [item_count] + self.encoder_dims + [latent_dim * 2]
        self.decoder_dims = [latent_dim] + self.decoder_dims + [item_count]

        self.encoder = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(
                    self.encoder_dims[:-1], self.encoder_dims[1:]
                )
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(
                    self.decoder_dims[:-1], self.decoder_dims[1:]
                )
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()

        for layer in self.encoder:
            self.weight_init(layer)

        for layer in self.decoder:
            self.weight_init(layer)

    def encode(self, batch: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Энкодер"""
        hidden = F.normalize(batch, p=2, dim=1)
        hidden = self.dropout(hidden)

        for layer in self.encoder[:-1]:
            hidden = layer(hidden)
            hidden = self.activation(hidden)

        hidden = self.encoder[-1](hidden)
        mu_latent = hidden[:, : self.latent_dim]
        logvar_latent = hidden[:, self.latent_dim :]
        return mu_latent, logvar_latent

    def reparameterize(
        self, mu_latent: torch.Tensor, logvar_latent: torch.Tensor
    ) -> torch.Tensor:
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
        for layer in self.decoder[:-1]:
            hidden = layer(hidden)
            hidden = self.activation(hidden)
        return self.decoder[-1](hidden)

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


class MultVAE(TorchRecommender):
    """
    Вариационный автокодировщик. Общая схема его работы
    представлена на рисунке.

    .. image:: /images/vae-gaussian.png

    **Постановка задачи**

    Дана выборка независимых одинаково распределенных величин из истинного
    распределения :math:`x_i \sim p_d(x)`, :math:`i = 1, \dots, N`.

    Задача - построить вероятностную модель :math:`p_\\theta(x)` истинного
    распределения :math:`p_d(x)`.

    Распределение :math:`p_\\theta(x)` должно позволять как оценить плотность
    вероятности для данного объекта :math:`x`, так и сэмплировать
    :math:`x \sim p_\\theta(x)`.

    **Вероятностная модель**

    :math:`z \in \mathbb{R}^d` - локальная латентная переменная, т. е. своя для
    каждого объекта :math:`x`.

    Генеративный процесс вариационного автокодировщика:

    1. Сэмплируем :math:`z \sim p(z)`.
    2. Сэмплируем :math:`x \sim p_\\theta(x | z)`.

    Параметры распределения :math:`p_\\theta(x | z)` задаются нейросетью с
    весами :math:`\\theta`, получающей на вход вектор :math:`z`.

    Индуцированная генеративным процессом плотность вероятности объекта
    :math:`x`:

    .. math::
        p_\\theta(x) = \mathbb{E}_{z \\sim p(z)} p_\\theta(x | z)

    В случачае ВАЕ для максимизации правдоподобия максимизируют вариационную
    нижнюю оценку на логарифм правдоподобия

    .. math::
        \log p_\\theta(x) = \mathbb{E}_{z \sim q_\phi(z | x)} \log p_\\theta(
        x) = \mathbb{E}_{z \sim q_\phi(z | x)} \log \\frac{p_\\theta(x,
        z) q_\phi(z | x)} {q_\phi(z | x) p_\\theta(z | x)} = \n
        = \mathbb{E}_{z
        \\sim q_\phi(z | x)} \log \\frac{p_\\theta(x, z)}{q_\phi(z | x)} + KL(
        q_\phi(z | x) || p_\\theta(z | x))

    .. math::
        \log p_\\theta(x) \geqslant \mathbb{E}_{z \sim q_\phi(z | x)}
        \log \\frac{p_\\theta(x | z)p(z)}{q_\phi(z | x)} =
        \mathbb{E}_{z \\sim q_\phi(z | x)} \log p_\\theta(x | z) -
        KL(q_\phi(z | x) || p(z)) = \n
        = L(x; \phi, \\theta) \\to \max\limits_{\phi, \\theta}

    :math:`q_\phi(z | x)` называется предложным (proposal) или распознающим
    (recognition) распределением. Это гауссиана, чьи параметры задаются
    нейросетью с весами :math:`\phi`:
    :math:`q_\phi(z | x) = \mathcal{N}(z | \mu_\phi(x), \sigma^2_\phi(x)I)`.

    Зазор между вариационной нижней оценкой :math:`L(x; \phi, \\theta)` на
    логарифм правдоподобия модели и самим логарифмом правдоподобия
    :math:`\log p_\\theta(x)` - это KL-дивергенция между предолжным и
    апостериорным распределением на :math:`z`:
    :math:`KL(q_\phi(z | x) || p_\\theta(z | x))`. Максимальное значение
    :math:`L(x; \phi, \\theta)` при фиксированных параметрах модели
    :math:`\\theta`
    достигается при :math:`q_\phi(z | x) = p_\\theta(z | x)`, но явное
    вычисление :math:`p_\\theta(z | x)` требует слишком большого числа
    ресурсов, поэтому вместо этого вычисления вариационная нижняя оценка
    оптимизируется также по :math:`\phi`. Чем ближе :math:`q_\phi(z | x)` к
    :math:`p_\\theta(z | x)`, тем точнее вариационная нижняя оценка.

    Обычно в качестве априорного распределения :math:`p(z)` используетя
    какое-то простое распределение, чаще всего нормальное:

    .. math::
        \\varepsilon \sim \mathcal{N}(\\varepsilon | 0, I)

    .. math::
        z = \mu + \sigma \\varepsilon \Rightarrow z \sim \mathcal{N}(z | \mu,
        \sigma^2I)

    .. math::
        \\frac{\partial}{\partial \phi} L(x; \phi, \\theta) = \mathbb{E}_{
        \\varepsilon \sim \mathcal{N}(\\varepsilon | 0, I)} \\frac{\partial}
        {\partial \phi} \log p_\\theta(x | \mu_\phi(x) + \sigma_\phi(x)
        \\varepsilon) - \\frac{\partial}{\partial \phi} KL(q_\phi(z | x) ||
        p(z))

    .. math::
        \\frac{\partial}{\partial \\theta} L(x; \phi, \\theta) = \mathbb{E}_{z
        \sim q_\phi(z | x)} \\frac{\partial}{\partial \\theta} \log
        p_\\theta(x | z)

    В этом случае

    .. math::
        KL(q_\phi(z | x) || p(z)) = -\\frac{1}{2}\sum_{i=1}^{dimZ}(1+
        log(\sigma_i^2) - \mu_i^2-\sigma_i^2)

    Также коэффициент при KL-дивергенции (коэффициент отжига) может быть
    положен не равным единице. Тогда оптимизируемая функция выглядит
    следующим образом

    .. math::
        L(x; \phi, \\theta) =
        \mathbb{E}_{z \\sim q_\phi(z | x)} \log p_\\theta(x | z) -
        \\beta \cdot KL(q_\phi(z | x) || p(z)) \\to \max\limits_{\phi, \\theta}

    При :math:`\\beta = 0` VAE (вариационный автокодировщик) превращается в
    DAE (шумоподавляющий автокодировщик)
    """

    num_workers: int = 0
    batch_size_users: int = 5000
    patience: int = 10
    n_saved: int = 2
    valid_split_size: float = 0.1
    seed: int = 42
    can_predict_cold_users = True

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 1,
        latent_dim: int = 10,
        decoder_dims: Optional[List[int]] = None,
        encoder_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        anneal: float = 0.005,
        l2_reg: float = 0,
        gamma: float = 0.99,
    ):
        """
        Инициализирует параметры модели и сохраняет спарк-сессию.

        :param learning_rate: шаг обучения
        :param epochs: количество эпох, в течение которых учимся
        :param latent_dim: размер скрытого представления пользователя
        :param decoder_dims: последовательность размеров скрытых слоев декодера
        :param encoder_dims: последовательность размеров скрытых слоев энкодера
        :param dropout: коэффициент дропаута
        :param anneal: коэффициент отжига от 0 до 1
        :param l2_reg: коэффициент l2 регуляризации
        :param gamma: коэффициент уменьшения learning_rate после каждой эпохи
        """
        self.device = State().device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.decoder_dims = decoder_dims
        self.encoder_dims = encoder_dims
        self.dropout = dropout
        self.anneal = anneal
        self.l2_reg = l2_reg
        self.gamma = gamma

    def get_params(self) -> Dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "latent_dim": self.latent_dim,
            "decoder_dims": self.decoder_dims,
            "encoder_dims": self.encoder_dims,
            "dropout": self.dropout,
            "anneal": self.anneal,
            "l2_reg": self.l2_reg,
            "gamma": self.gamma,
        }

    def _get_data_loader(
        self, data: pd.DataFrame, shuffle: bool = True
    ) -> (csr_matrix, DataLoader, np.array):
        """Функция получения загрузчика данных, а также матрицы с данными"""
        users_count = data["user_idx"].value_counts().count()
        user_idx = data.user_idx.astype("category").cat
        user_batch = csr_matrix(
            (
                np.ones(len(data.user_idx)),
                ([user_idx.codes.values, data.item_idx.values]),
            ),
            shape=(users_count, self.items_count),
        )
        data_loader = DataLoader(
            TensorDataset(torch.arange(users_count).long()),
            batch_size=self.batch_size_users,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

        return user_batch, data_loader, user_idx.categories.values

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.logger.debug("Индексирование данных")
        log_indexed = self.user_indexer.transform(log)
        log_indexed = self.item_indexer.transform(log_indexed)

        self.logger.debug("Составление батча:")
        data = log_indexed.select("user_idx", "item_idx").toPandas()
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=self.valid_split_size, random_state=self.seed
        )
        train_idx, valid_idx = next(
            splitter.split(data, groups=data["user_idx"])
        )
        train_data, valid_data = data.iloc[train_idx], data.iloc[valid_idx]

        self.train_user_batch, train_data_loader, _ = self._get_data_loader(
            train_data
        )
        self.valid_user_batch, valid_data_loader, _ = self._get_data_loader(
            valid_data, False
        )

        self.logger.debug("Обучение модели")
        self.model = VAE(
            item_count=self.items_count,
            latent_dim=self.latent_dim,
            decoder_dims=self.decoder_dims,
            encoder_dims=self.encoder_dims,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg / self.batch_size_users,
        )
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

        vae_trainer, val_evaluator = self._create_trainer_evaluator(
            optimizer,
            valid_data_loader,
            lr_scheduler,
            self.patience,
            self.n_saved,
        )

        vae_trainer.run(train_data_loader, max_epochs=self.epochs)

    def _loss(self, y_pred, y_true, mu_latent, logvar_latent):
        log_softmax_var = F.log_softmax(y_pred, dim=1)
        bce = -(log_softmax_var * y_true).sum(dim=1).mean()
        kld = (
            -0.5
            * torch.sum(
                1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp(),
                dim=1,
            ).mean()
        )
        return bce + self.anneal * kld

    def _batch_pass(self, batch, model):
        if model.training:
            full_batch = self.train_user_batch
        else:
            full_batch = self.valid_user_batch
        user_batch = torch.FloatTensor(full_batch[batch[0]].toarray()).to(
            self.device
        )
        pred_user_batch, latent_mu, latent_logvar = self.model(user_batch)
        return (
            pred_user_batch,
            user_batch,
            {"mu_latent": latent_mu, "logvar_latent": latent_logvar},
        )

    @staticmethod
    def _predict_by_user(
        pandas_df: pd.DataFrame,
        model: nn.Module,
        items_np: np.array,
        k: int,
        items_count: int,
    ) -> pd.DataFrame:
        user_idx = pandas_df["user_idx"][0]
        cnt = min(len(pandas_df) + k, len(items_np))

        model.eval()
        with torch.no_grad():
            user_batch = torch.zeros((1, items_count))
            user_batch[0, pandas_df["item_idx"].values] = 1
            user_recs = model(user_batch)[0][0].detach()
            best_item_idx = (
                torch.argsort(user_recs[items_np], descending=True)[:cnt]
            ).numpy()

            return pd.DataFrame(
                {
                    "user_idx": cnt * [user_idx],
                    "item_idx": items_np[best_item_idx],
                    "relevance": user_recs[best_item_idx],
                }
            )
