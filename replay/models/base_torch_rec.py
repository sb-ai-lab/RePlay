from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from ignite.contrib.handlers import LRScheduler
from ignite.engine import Engine, Events
from ignite.handlers import (
    EarlyStopping,
    ModelCheckpoint,
    global_step_from_engine,
)
from ignite.metrics import Loss, RunningAverage
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from torch import nn
from torch.optim.optimizer import Optimizer  # pylint: disable=E0611
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader

from replay.models.base_rec import Recommender
from replay.session_handler import State
from replay.constants import IDX_REC_SCHEMA


class TorchRecommender(Recommender):
    """Base class for neural recommenders"""

    model: Any
    device: torch.device

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        items_pd = items.toPandas()["item_idx"].values
        items_count = self.items_count
        model = self.model.cpu()
        agg_fn = self._predict_by_user

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            return agg_fn(pandas_df, model, items_pd, k, items_count)[
                ["user_idx", "item_idx", "relevance"]
            ]

        self.logger.debug("Предсказание модели")
        recs = (
            users.join(log, how="left", on="user_idx")
            .selectExpr(
                "user_idx AS user_idx",
                "item_idx AS item_idx",
            )
            .groupby("user_idx")
            .applyInPandas(grouped_map, IDX_REC_SCHEMA)
        )
        return recs

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        items_count = self.items_count
        model = self.model.cpu()
        agg_fn = self._predict_by_user_pairs
        users = pairs.select("user_idx").distinct()

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            return agg_fn(pandas_df, model, items_count)[
                ["user_idx", "item_idx", "relevance"]
            ]

        self.logger.debug("Оценка релевантности для пар")
        user_history = (
            users.join(log, how="inner", on="user_idx")
            .groupBy("user_idx")
            .agg(sf.collect_list("item_idx").alias("item_idx_history"))
        )
        user_pairs = pairs.groupBy("user_idx").agg(
            sf.collect_list("item_idx").alias("item_idx_to_pred")
        )
        full_df = user_pairs.join(user_history, on="user_idx", how="left")

        recs = full_df.groupby("user_idx").applyInPandas(
            grouped_map, IDX_REC_SCHEMA
        )

        return recs

    @staticmethod
    @abstractmethod
    def _predict_by_user(
        pandas_df: pd.DataFrame,
        model: nn.Module,
        items_np: np.ndarray,
        k: int,
        item_count: int,
    ) -> pd.DataFrame:
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
        pandas_df: pd.DataFrame,
        model: nn.Module,
        item_count: int,
    ) -> pd.DataFrame:
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

    # pylint: disable=too-many-arguments
    def _create_trainer_evaluator(
        self,
        opt: Optimizer,
        valid_data_loader: DataLoader,
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
        early_stopping_patience: Optional[int] = None,
        checkpoint_number: Optional[int] = None,
    ) -> Tuple[Engine, Engine]:
        """
        Creates a trainer and en evaluator needed to train model

        :param opt: optimizer
        :param valid_data_loader: data loader for validation
        :param scheduler: scheduler used to decrease learning rate
        :param early_stopping_patience: number of epochs used for early stopping
        :param checkpoint_number: number of best checkpoints
        :return: trainer, evaluator
        """
        self.model.to(self.device)  # pylint: disable=E1101

        # pylint: disable=unused-argument
        def _run_train_step(engine, batch):
            self.model.train()
            opt.zero_grad()
            model_result = self._batch_pass(batch, self.model)
            y_pred, y_true = model_result[:2]
            if len(model_result) == 2:
                loss = self._loss(y_pred, y_true)
            else:
                loss = self._loss(y_pred, y_true, **model_result[2])
            loss.backward()
            opt.step()
            return loss.item()

        # pylint: disable=unused-argument
        def _run_val_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                return self._batch_pass(batch, self.model)

        torch_trainer = Engine(_run_train_step)
        torch_evaluator = Engine(_run_val_step)

        avg_output = RunningAverage(output_transform=lambda x: x)
        avg_output.attach(torch_trainer, "loss")
        Loss(self._loss).attach(torch_evaluator, "loss")

        # pylint: disable=unused-variable
        @torch_trainer.on(Events.EPOCH_COMPLETED)
        def log_training_loss(trainer):
            debug_message = f"""Epoch[{trainer.state.epoch}] current loss:
{trainer.state.metrics["loss"]:.5f}"""
            self.logger.debug(debug_message)

        # pylint: disable=unused-variable
        @torch_trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            torch_evaluator.run(valid_data_loader)
            metrics = torch_evaluator.state.metrics
            debug_message = f"""Epoch[{trainer.state.epoch}] validation
average loss: {metrics["loss"]:.5f}"""
            self.logger.debug(debug_message)

        def score_function(engine):
            return -engine.state.metrics["loss"]

        if early_stopping_patience:
            self._add_early_stopping(
                early_stopping_patience,
                score_function,
                torch_trainer,
                torch_evaluator,
            )
        if checkpoint_number:
            self._add_checkpoint(
                checkpoint_number,
                score_function,
                torch_trainer,
                torch_evaluator,
            )
        if scheduler:
            self._add_scheduler(scheduler, torch_trainer, torch_evaluator)

        return torch_trainer, torch_evaluator

    @staticmethod
    def _add_early_stopping(
        early_stopping_patience, score_function, torch_trainer, torch_evaluator
    ):
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            score_function=score_function,
            trainer=torch_trainer,
        )
        torch_evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    def _add_checkpoint(
        self, checkpoint_number, score_function, torch_trainer, torch_evaluator
    ):
        checkpoint = ModelCheckpoint(
            State().session.conf.get("spark.local.dir"),
            create_dir=True,
            require_empty=False,
            n_saved=checkpoint_number,
            score_function=score_function,
            score_name="loss",
            filename_prefix="best",
            global_step_transform=global_step_from_engine(torch_trainer),
        )

        torch_evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            checkpoint,
            {type(self).__name__.lower(): self.model},
        )

        # pylint: disable=unused-argument,unused-variable
        @torch_trainer.on(Events.COMPLETED)
        def load_best_model(engine):
            self.load_model(checkpoint.last_checkpoint)

    @staticmethod
    def _add_scheduler(scheduler, torch_trainer, torch_evaluator):
        if isinstance(scheduler, _LRScheduler):
            torch_trainer.add_event_handler(
                Events.EPOCH_COMPLETED, LRScheduler(scheduler)
            )
        else:

            @torch_evaluator.on(Events.EPOCH_COMPLETED)
            # pylint: disable=unused-variable
            def reduct_step(engine):
                scheduler.step(engine.state.metrics["loss"])

    @abstractmethod
    def _batch_pass(
        self, batch, model
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[None, Dict[str, Any]]]:
        """
        Apply model to a single batch.

        :param batch: data batch
        :param model: model object
        :return: y_pred, y_true, and a dictionary used to calculate loss.
        """

    @abstractmethod
    def _loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Returns loss value

        :param y_pred: output from model
        :param y_true: actual test data
        :param *args: other arguments used to calculate loss
        :return: 1x1 tensor
        """
