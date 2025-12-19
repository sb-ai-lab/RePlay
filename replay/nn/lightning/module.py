from typing import Any, Optional, Union

import lightning
import torch

from replay.models.nn.optimizer_utils import (
    FatOptimizerFactory,
    LRSchedulerFactory,
    OptimizerFactory,
)
from replay.nn import InferenceOutput, TrainOutput


class LightningModule(lightning.LightningModule):
    """
    A universal wrapper class above the PyTorch model for working with Lightning library.\n
    Pay attention to the format of the ``forward`` function's return value.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_factory: Optional[OptimizerFactory] = None,
        lr_scheduler_factory: Optional[LRSchedulerFactory] = None,
    ) -> None:
        """
        :param model: Initialized model.\n
            Expected result of the model's ``forward`` function
            is an object of the ``TrainOutput`` class after training stage
            and ``InferenceOutput`` after inference stage.
        :param optimizer_factory: Optimizer factory.
            Default: ``None``.
        :param lr_scheduler_factory: Learning rate schedule factory.
            Default: ``None``.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self._optimizer_factory = optimizer_factory
        self._lr_scheduler_factory = lr_scheduler_factory
        self.candidates_to_score = None

    def forward(self, batch: dict) -> Union[TrainOutput, InferenceOutput]:
        """
        Implementation of the forward function.

        :param batch: A dictionary containing all the necessary information to run the forward function on the model.
            The dictionary keys must match the names of the arguments in the model's forward function.
            Keys that do not match the arguments of the model's forward function are filtered out.
            If the model supports calculating logits for custom candidates on the inference stage,
            then you can submit them inside the batch or using the ``candidates_to_score`` field.
        :returns: During training, the model will return an object
            of the ``TrainOutput`` container class or its successor.
            At the inference stage, the ``InferenceOutput`` class or its successor will be returned.
        """
        if "candidates_to_score" not in batch and self.candidates_to_score is not None and not self.training:
            batch["candidates_to_score"] = self.candidates_to_score
        # select only args for model.forward
        modified_batch = {k: v for k, v in batch.items() if k in self.model.forward.__code__.co_varnames}
        return self.model(**modified_batch)

    def training_step(self, batch: dict) -> torch.Tensor:
        model_output: TrainOutput = self(batch)
        loss = model_output["loss"]
        lr = self.optimizers().param_groups[0]["lr"]  # Get current learning rate
        self.log("learning_rate", lr, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch: dict) -> torch.Tensor:
        model_output: InferenceOutput = self(batch)
        return model_output

    def validation_step(self, batch: dict) -> torch.Tensor:
        model_output: InferenceOutput = self(batch)
        return model_output

    def configure_optimizers(self) -> Any:
        """
        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
                Configured optimizer and lr scheduler.
        """
        optimizer_factory = self._optimizer_factory or FatOptimizerFactory()
        optimizer = optimizer_factory.create(self.model.parameters())

        if self._lr_scheduler_factory is None:
            return optimizer

        lr_scheduler = self._lr_scheduler_factory.create(optimizer)
        return [optimizer], [lr_scheduler]

    @property
    def candidates_to_score(self) -> Optional[torch.LongTensor]:
        """
        :getter: Returns a tensor containing the candidate IDs.
            The tendor will be used during the inference stage of the model.\n
            If the parameter was not previously set, ``None`` will be returned.
        :setter: A one-dimensional tensor containing candidate IDs is expected.
        """
        return self._candidates_to_score

    @candidates_to_score.setter
    def candidates_to_score(self, candidates: Optional[torch.LongTensor] = None) -> None:
        self._candidates_to_score = candidates
