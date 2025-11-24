from typing import Any, Optional, Union

import lightning
import torch

from replay.models.nn.optimizer_utils import FatOptimizerFactory, LRSchedulerFactory, OptimizerFactory

from .output import InferenceOutput, TrainOutput


class LightingModule(lightning.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_factory: Optional[OptimizerFactory] = None,
        lr_scheduler_factory: Optional[LRSchedulerFactory] = None,
    ):
        """
        Args:
            model (torch.nn.Module): initialized model.
            optimizer_factory (Optional[OptimizerFactory]): Optimizer factory.
                Default: ``None``.
            lr_scheduler_factory (Optional[LRSchedulerFactory]): Learning rate schedule factory.
                Default: ``None``.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self._optimizer_factory = optimizer_factory
        self._lr_scheduler_factory = lr_scheduler_factory
        self.candidates_to_score = None

    def forward(self, batch: dict) -> Union[TrainOutput, InferenceOutput]:
        if "candidates_to_score" not in batch and self.candidates_to_score is not None and not self.training:
            batch["candidates_to_score"] = self.candidates_to_score
        # select only args for model.forward
        modified_batch = {k: v for k, v in batch.items() if k in self.model.forward.__code__.co_varnames}
        return self.model(**modified_batch)

    def training_step(self, batch: dict) -> torch.Tensor:
        model_output: TrainOutput = self(batch)
        loss = model_output.loss
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
        return model_output.logits

    def validation_step(self, batch: dict) -> torch.Tensor:
        model_output: InferenceOutput = self(batch)
        return model_output.logits

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
    def candidates_to_score(self) -> Union[torch.LongTensor, None]:
        """
        Returns tensor of item ids to calculate scores.
        """
        return self._candidates_to_score

    @candidates_to_score.setter
    def candidates_to_score(self, candidates: Optional[torch.LongTensor] = None) -> None:
        """
        Sets tensor of item ids to calculate scores.
        :param candidates: Tensor of item ids to calculate scores.
        """
        self._candidates_to_score = candidates
