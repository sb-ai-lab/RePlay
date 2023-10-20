"""
MultVAE implementation
(Variational Autoencoders for Collaborative Filtering)
"""
from typing import Optional, Tuple

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

from replay.experimental.models.base_torch_rec import TorchRecommender


class VAE(nn.Module):
    """Base variational autoencoder"""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        item_count: int,
        latent_dim: int,
        hidden_dim: int = 600,
        dropout: float = 0.3,
    ):
        """
        :param item_count: number of items
        :param latent_dim: latent dimension size
        :param hidden_dim: hidden dimension size for encoder and decoder
        :param dropout: dropout coefficient
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder_dims = [item_count, hidden_dim, latent_dim * 2]
        self.decoder_dims = [latent_dim, hidden_dim, item_count]

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

    def encode(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode"""
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
        """Reparametrization trick"""

        if self.training:
            std = torch.exp(0.5 * logvar_latent)
            eps = torch.randn_like(std)
            return eps * std + mu_latent
        return mu_latent

    def decode(self, z_latent: torch.Tensor) -> torch.Tensor:
        """Decode"""
        hidden = z_latent
        for layer in self.decoder[:-1]:
            hidden = layer(hidden)
            hidden = self.activation(hidden)
        return self.decoder[-1](hidden)  # type: ignore

    # pylint: disable=arguments-differ
    def forward(  # type: ignore
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param batch: user batch
        :return: output, expectation and logarithm of variation
        """
        mu_latent, logvar_latent = self.encode(batch)
        z_latent = self.reparameterize(mu_latent, logvar_latent)
        return self.decode(z_latent), mu_latent, logvar_latent

    @staticmethod
    def weight_init(layer: nn.Module):
        """
        Xavier initialization

        :param layer: layer of a model
        """
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight.data)
            layer.bias.data.normal_(0.0, 0.001)


# pylint: disable=too-many-instance-attributes
class MultVAE(TorchRecommender):
    """`Variational Autoencoders for Collaborative Filtering
    <https://arxiv.org/pdf/1802.05814.pdf>`_"""

    num_workers: int = 0
    batch_size_users: int = 5000
    patience: int = 10
    n_saved: int = 2
    valid_split_size: float = 0.1
    seed: int = 42
    can_predict_cold_users = True
    train_user_batch: csr_matrix
    valid_user_batch: csr_matrix
    _search_space = {
        "learning_rate": {"type": "loguniform", "args": [0.0001, 0.5]},
        "epochs": {"type": "int", "args": [100, 100]},
        "latent_dim": {"type": "int", "args": [200, 200]},
        "hidden_dim": {"type": "int", "args": [600, 600]},
        "dropout": {"type": "uniform", "args": [0, 0.5]},
        "anneal": {"type": "uniform", "args": [0.2, 1]},
        "l2_reg": {"type": "loguniform", "args": [1e-9, 5]},
        "factor": {"type": "uniform", "args": [0.2, 0.2]},
        "patience": {"type": "int", "args": [3, 3]},
    }

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 100,
        latent_dim: int = 200,
        hidden_dim: int = 600,
        dropout: float = 0.3,
        anneal: float = 0.1,
        l2_reg: float = 0,
        factor: float = 0.2,
        patience: int = 3,
    ):
        """
        :param learning_rate: learning rate
        :param epochs: number of epochs to train model
        :param latent_dim: latent dimension size for user vectors
        :param hidden_dim: hidden dimension size for encoder and decoder
        :param dropout: dropout coefficient
        :param anneal: anneal coefficient [0,1]
        :param l2_reg: l2 regularization term
        :param factor: ReduceLROnPlateau reducing factor. new_lr = lr * factor
        :param patience: number of non-improved epochs before reducing lr
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.anneal = anneal
        self.l2_reg = l2_reg
        self.factor = factor
        self.patience = patience

    @property
    def _init_args(self):
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "anneal": self.anneal,
            "l2_reg": self.l2_reg,
            "factor": self.factor,
            "patience": self.patience,
        }

    def _get_data_loader(
        self, data: pd.DataFrame, shuffle: bool = True
    ) -> Tuple[csr_matrix, DataLoader, np.ndarray]:
        """get data loader and matrix with data"""
        users_count = data["user_idx"].value_counts().count()
        user_idx = data["user_idx"].astype("category").cat  # type: ignore
        user_batch = csr_matrix(
            (
                np.ones(len(data["user_idx"])),
                ([user_idx.codes.values, data["item_idx"].values]),
            ),
            shape=(users_count, self._item_dim),
        )
        data_loader = DataLoader(
            TensorDataset(torch.arange(users_count).long()),
            batch_size=self.batch_size_users,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

        return user_batch, data_loader, user_idx.categories.values

    # pylint: disable=too-many-locals
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.logger.debug("Creating batch")
        data = log.select("user_idx", "item_idx").toPandas()
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

        self.logger.debug("Training VAE")
        self.model = VAE(
            item_count=self._item_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg / self.batch_size_users,
        )
        lr_scheduler = ReduceLROnPlateau(
            optimizer, factor=self.factor, patience=self.patience
        )

        self.train(
            train_data_loader,
            valid_data_loader,
            optimizer,
            lr_scheduler,
            self.epochs,
            "multvae",
        )

    # pylint: disable=arguments-differ
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
        pred_user_batch, latent_mu, latent_logvar = self.model.forward(
            user_batch
        )
        return {
            "y_pred": pred_user_batch,
            "y_true": user_batch,
            "mu_latent": latent_mu,
            "logvar_latent": latent_logvar,
        }

    @staticmethod
    def _predict_pairs_inner(
        model: nn.Module,
        user_idx: int,
        items_np_history: np.ndarray,
        items_np_to_pred: np.ndarray,
        item_count: int,
        cnt: Optional[int] = None,
    ) -> DataFrame:
        model.eval()
        with torch.no_grad():
            user_batch = torch.zeros((1, item_count))
            user_batch[0, items_np_history] = 1
            user_recs = F.softmax(model(user_batch)[0][0].detach(), dim=0)
            if cnt is not None:
                best_item_idx = (
                    torch.argsort(
                        user_recs[items_np_to_pred], descending=True
                    )[:cnt]
                ).numpy()
                items_np_to_pred = items_np_to_pred[best_item_idx]
            return pd.DataFrame(
                {
                    "user_idx": np.array(
                        items_np_to_pred.shape[0] * [user_idx]
                    ),
                    "item_idx": items_np_to_pred,
                    "relevance": user_recs[items_np_to_pred],
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
        return MultVAE._predict_pairs_inner(
            model=model,
            user_idx=pandas_df["user_idx"][0],
            items_np_history=pandas_df["item_idx"].values,
            items_np_to_pred=items_np,
            item_count=item_count,
            cnt=min(len(pandas_df) + k, len(items_np)),
        )

    @staticmethod
    def _predict_by_user_pairs(
        pandas_df: pd.DataFrame,
        model: nn.Module,
        item_count: int,
    ) -> pd.DataFrame:
        return MultVAE._predict_pairs_inner(
            model=model,
            user_idx=pandas_df["user_idx"][0],
            items_np_history=np.array(pandas_df["item_idx_history"][0]),
            items_np_to_pred=np.array(pandas_df["item_idx_to_pred"][0]),
            item_count=item_count,
            cnt=None,
        )

    def _load_model(self, path: str):
        self.model = VAE(
            item_count=self._item_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
