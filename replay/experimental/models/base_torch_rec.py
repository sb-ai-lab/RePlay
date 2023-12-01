from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from replay.data import get_schema
from replay.experimental.models.base_rec import Recommender
from replay.experimental.utils.session_handler import State
from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, SparkDataFrame

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf


class TorchRecommender(Recommender):
    """Base class for neural recommenders"""

    model: Any
    device: torch.device

    def __init__(self):
        self.logger.info(
            "The model is neural network with non-distributed training"
        )
        self.checkpoint_path = State().session.conf.get("spark.local.dir")
        self.device = State().device

    def _run_train_step(self, batch, optimizer):
        self.model.train()
        optimizer.zero_grad()
        model_result = self._batch_pass(batch, self.model)
        loss = self._loss(**model_result)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _run_validation(
        self, valid_data_loader: DataLoader, epoch: int
    ) -> float:
        self.model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_data_loader:
                model_result = self._batch_pass(batch, self.model)
                valid_loss += self._loss(**model_result)
            valid_loss /= len(valid_data_loader)
            valid_debug_message = f"""Epoch[{epoch}] validation
                                    average loss: {valid_loss:.5f}"""
            self.logger.debug(valid_debug_message)
        return valid_loss.item()

    # pylint: disable=too-many-arguments
    def train(
        self,
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader,
        optimizer: Optimizer,
        lr_scheduler: ReduceLROnPlateau,
        epochs: int,
        model_name: str,
    ) -> None:
        """
        Run training loop
        :param train_data_loader: data loader for training
        :param valid_data_loader: data loader for validation
        :param optimizer: optimizer
        :param lr_scheduler: scheduler used to decrease learning rate
        :param lr_scheduler: scheduler used to decrease learning rate
        :param epochs: num training epochs
        :param model_name: model name for checkpoint saving
        :return:
        """
        best_valid_loss = np.inf
        for epoch in range(epochs):
            for batch in train_data_loader:
                train_loss = self._run_train_step(batch, optimizer)

            train_debug_message = f"""Epoch[{epoch}] current loss:
                                    {train_loss:.5f}"""
            self.logger.debug(train_debug_message)

            valid_loss = self._run_validation(valid_data_loader, epoch)
            lr_scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_checkpoint = "/".join(
                    [
                        self.checkpoint_path,
                        f"/best_{model_name}_{epoch+1}_loss={valid_loss}.pt",
                    ]
                )
                self._save_model(best_checkpoint)
                best_valid_loss = valid_loss
        self._load_model(best_checkpoint)

    @abstractmethod
    def _batch_pass(self, batch, model) -> Dict[str, Any]:
        """
        Apply model to a single batch.

        :param batch: data batch
        :param model: model object
        :return: dictionary used to calculate loss.
        """

    @abstractmethod
    def _loss(self, **kwargs) -> torch.Tensor:
        """
        Returns loss value

        :param **kwargs: dictionary used to calculate loss
        :return: 1x1 tensor
        """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
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
        items_count = self._item_dim
        model = self.model.cpu()
        agg_fn = self._predict_by_user

        def grouped_map(pandas_df: PandasDataFrame) -> PandasDataFrame:
            return agg_fn(
                pandas_df, model, items_consider_in_pred, k, items_count
            )[["user_idx", "item_idx", "relevance"]]

        self.logger.debug("Predict started")
        # do not apply map on cold users for MultVAE predict
        join_type = "inner" if str(self) == "MultVAE" else "left"
        rec_schema = get_schema(
            query_column="user_idx",
            item_column="item_idx",
            rating_column="relevance",
            has_timestamp=False,
        )
        recs = (
            users.join(log, how=join_type, on="user_idx")
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
        items_count = self._item_dim
        model = self.model.cpu()
        agg_fn = self._predict_by_user_pairs
        users = pairs.select("user_idx").distinct()

        def grouped_map(pandas_df: PandasDataFrame) -> PandasDataFrame:
            return agg_fn(pandas_df, model, items_count)[
                ["user_idx", "item_idx", "relevance"]
            ]

        self.logger.debug("Calculate relevance for user-item pairs")
        user_history = (
            users.join(log, how="inner", on="user_idx")
            .groupBy("user_idx")
            .agg(sf.collect_list("item_idx").alias("item_idx_history"))
        )
        user_pairs = pairs.groupBy("user_idx").agg(
            sf.collect_list("item_idx").alias("item_idx_to_pred")
        )
        full_df = user_pairs.join(user_history, on="user_idx", how="inner")

        rec_schema = get_schema(
            query_column="user_idx",
            item_column="item_idx",
            rating_column="relevance",
            has_timestamp=False,
        )
        recs = full_df.groupby("user_idx").applyInPandas(
            grouped_map, rec_schema
        )

        return recs

    @staticmethod
    @abstractmethod
    def _predict_by_user(
        pandas_df: PandasDataFrame,
        model: nn.Module,
        items_np: np.ndarray,
        k: int,
        item_count: int,
    ) -> PandasDataFrame:
        """
        Calculate predictions.

        :param pandas_df: DataFrame with user-item interactions ``[user_idx, item_idx]``
        :param model: trained model
        :param items_np: items available for recommendations
        :param k: length of recommendation list
        :param item_count: total number of items
        :return: DataFrame ``[user_idx , item_idx , relevance]``
        """

    @staticmethod
    @abstractmethod
    def _predict_by_user_pairs(
        pandas_df: PandasDataFrame,
        model: nn.Module,
        item_count: int,
    ) -> PandasDataFrame:
        """
        Get relevance for provided pairs

        :param pandas_df: DataFrame with rated items and items that need prediction
            ``[user_idx, item_idx_history, item_idx_to_pred]``
        :param model: trained model
        :param item_count: total number of items
        :return: DataFrame ``[user_idx , item_idx , relevance]``
        """

    def load_model(self, path: str) -> None:
        """
        Load model from file

        :param path: path to model
        :return:
        """
        self.logger.debug("-- Loading model from file")
        self.model.load_state_dict(torch.load(path))

    def _save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
