import inspect
from collections.abc import Mapping
from typing import Any

import lightning
import torch
from typing_extensions import override

from replay.nn.lightning.optimizer import BaseOptimizerFactory, OptimizerFactory
from replay.nn.lightning.scheduler import BaseLRSchedulerFactory
from replay.nn.output import InferenceOutput, TrainOutput


class LightningModule(lightning.LightningModule):
    """
    A universal wrapper class above the PyTorch model for working with the Lightning library.\n
    Pay attention to the format of the ``forward`` function's return value.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_factory: BaseOptimizerFactory | None = None,
        lr_scheduler_factory: BaseLRSchedulerFactory | None = None,
        *,
        epoch_only_training_metrics: bool = False,
        training_batch_size_feature_name: str | None = None,
    ) -> None:
        """
        :param model: Initialized model.\n
            The expected result of the ``forward`` function
            is an object of the ``TrainOutput`` class after the training stage
            and `InferenceOutput` after the inference stage.
        :param optimizer_factory: The optimizer factory.
            Default: ``None``.
        :param lr_scheduler_factory: The learning rate schedule factory.
            Default: ``None``.
        :param epoch_only_training_metrics: accumulate training metrics locally and synchronize them
            at epoch end instead of every step. Default: ``False``.
        :param training_batch_size_feature_name: feature used to determine batch size when
            epoch-level reduction is enabled. By default all tensors in ``feature_tensors`` are checked.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self._optimizer_factory = optimizer_factory
        self._lr_scheduler_factory = lr_scheduler_factory
        self.candidates_to_score = None
        self.epoch_only_training_metrics = epoch_only_training_metrics
        self.training_batch_size_feature_name = training_batch_size_feature_name
        # Keep this outside registered buffers: DDP broadcasts buffers before each forward pass.
        self._training_metric_totals: torch.Tensor | None = None

    def forward(self, batch: dict) -> TrainOutput | InferenceOutput:
        """
        Implementation of the forward function.

        :param batch: A dictionary containing all the necessary information to run the forward function on the model.
            The dictionary keys must match the names of the arguments in the model's forward function.
            Keys that do not match the arguments of the model's forward function are filtered out.
            If the model supports calculating logits for custom candidates at the inference stage,
            then you can submit them inside the batch or using the ``candidates_to_score`` field.
        :returns: During training, the model will return an object
            of the ``TrainOutput`` container class or its successor.
            At the inference stage, the ``InferenceOutput`` class or its successor will be returned.
        """
        if "candidates_to_score" not in batch and self.candidates_to_score is not None and not self.training:
            batch["candidates_to_score"] = self.candidates_to_score
        # select only args for model.forward
        modified_batch = {k: v for k, v in batch.items() if k in inspect.signature(self.model.forward).parameters}
        return self.model(**modified_batch)

    def training_step(self, batch: dict) -> torch.Tensor:
        if self.epoch_only_training_metrics:
            return self._training_step_with_epoch_metrics(batch)
        model_output: TrainOutput = self(batch)
        loss = model_output["loss"]
        lr = self.optimizers().param_groups[0]["lr"]  # Get current learning rate
        self.log("learning_rate", lr, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def _infer_training_batch_size(self, batch: Mapping[str, Any]) -> int:
        feature_tensors = batch.get("feature_tensors")
        batch_tensors = feature_tensors if isinstance(feature_tensors, Mapping) else batch
        feature_name = self.training_batch_size_feature_name
        if feature_name is not None:
            feature = batch_tensors.get(feature_name)
            if not isinstance(feature, torch.Tensor) or feature.ndim == 0:
                msg = f"Batch-size feature '{feature_name}' must be a non-scalar tensor."
                raise ValueError(msg)
            batch_sizes = {int(feature.shape[0])}
        else:
            batch_sizes = {
                int(value.shape[0])
                for value in batch_tensors.values()
                if isinstance(value, torch.Tensor) and value.ndim > 0
            }
            if len(batch_sizes) != 1:
                msg = (
                    "Batch tensors must have one common batch size. Set "
                    "training_batch_size_feature_name when the mapping contains shared tensors."
                )
                raise ValueError(msg)
        batch_size = batch_sizes.pop()
        if batch_size <= 0:
            msg = f"Training batch size must be positive, got {batch_size}."
            raise ValueError(msg)
        return batch_size

    def _training_step_with_epoch_metrics(self, batch: dict) -> torch.Tensor:
        model_output: TrainOutput = self(batch)
        loss = model_output["loss"]
        learning_rate = self.optimizers().param_groups[0]["lr"]
        batch_size = self._infer_training_batch_size(batch)
        if self._training_metric_totals is None:
            self._training_metric_totals = torch.zeros(3, dtype=torch.float64, device=loss.device)
        totals = self._training_metric_totals
        detached_loss = loss.detach().to(device=totals.device, dtype=totals.dtype)
        detached_learning_rate = torch.as_tensor(
            learning_rate,
            device=totals.device,
            dtype=totals.dtype,
        )
        totals[0].add_(detached_loss * batch_size)
        totals[1].add_(detached_learning_rate * batch_size)
        totals[2].add_(batch_size)

        if self.global_rank == 0:
            self.log(
                "learning_rate_step",
                detached_learning_rate,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=False,
                rank_zero_only=True,
                batch_size=batch_size,
            )
            self.log(
                "train_loss_step",
                detached_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=False,
                rank_zero_only=True,
                batch_size=batch_size,
            )

        return loss

    def on_train_epoch_start(self) -> None:
        if self.epoch_only_training_metrics:
            self._training_metric_totals = torch.zeros(3, dtype=torch.float64, device=self.device)
        super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        if self.epoch_only_training_metrics:
            totals = self._training_metric_totals
            if totals is None:
                msg = "Cannot aggregate training metrics without rows."
                raise RuntimeError(msg)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
            if totals[2].item() <= 0:
                msg = "Cannot aggregate training metrics without rows."
                raise RuntimeError(msg)
            for name, value in (
                ("train_loss_epoch", totals[0] / totals[2]),
                ("learning_rate_epoch", totals[1] / totals[2]),
            ):
                self.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=False,
                )
        super().on_train_epoch_end()

    @override
    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        model_output: InferenceOutput = self(batch)
        return model_output

    @override
    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        model_output: InferenceOutput = self(batch)
        return model_output

    @override
    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        model_output: InferenceOutput = self(batch)
        return model_output

    def configure_optimizers(self) -> Any:
        """
        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
                Configured optimizer and lr scheduler.
        """
        optimizer_factory = self._optimizer_factory or OptimizerFactory()
        optimizer = optimizer_factory.create(self.model.parameters())

        if self._lr_scheduler_factory is None:
            return optimizer

        lr_scheduler = self._lr_scheduler_factory.create(optimizer)
        return [optimizer], [lr_scheduler]

    @property
    def candidates_to_score(self) -> torch.LongTensor | None:
        """
        :getter: Returns a tensor containing the candidate IDs.
            The tensor will be used during the inference stage of the model.\n
            If the parameter was not previously set, ``None`` will be returned.
        :setter: A one-dimensional tensor containing candidate IDs is expected.
        """
        return self._candidates_to_score

    @candidates_to_score.setter
    def candidates_to_score(self, candidates: torch.LongTensor | None = None) -> None:
        if (candidates is not None) and bool(candidates.unique().numel() != candidates.numel()):
            msg = "The tensor of candidates to score must be unique."
            raise ValueError(msg)

        self._candidates_to_score = candidates
