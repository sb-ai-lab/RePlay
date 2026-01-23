from typing import Any, Optional

import lightning
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from replay.metrics.torch_metrics_builder import (
    MetricName,
    TorchMetricsBuilder,
    metrics_to_df,
)
from replay.nn.lightning import LightningModule
from replay.nn.lightning.postprocessor import PostprocessorBase
from replay.nn.output import InferenceOutput


class ComputeMetricsCallback(lightning.Callback):
    """
    Callback for validation and testing stages.

    If multiple validation/testing dataloaders are used,
    the suffix of the metric name will contain the serial number of the dataloader.

    For the correct calculation of metrics inside the callback,
    the batch must contain the ``ground_truth_column`` key - the padding value of this tensor can be any,
    the main condition is that the padding value does not overlap with the existing item ID values.
    For example, these can be negative values.

    To calculate the ``coverage`` and ``novelty`` metrics, the batch must additionally contain the ``train_column`` key.
    The padding value of this tensor can be any, the main condition is that the padding value does not overlap
    with the existing item ID values. For example, these can be negative values.
    """

    def __init__(
        self,
        metrics: Optional[list[MetricName]] = None,
        ks: Optional[list[int]] = None,
        postprocessors: Optional[list[PostprocessorBase]] = None,
        item_count: Optional[int] = None,
        ground_truth_column: str = "ground_truth",
        train_column: str = "train",
    ):
        """
        :param metrics: Sequence of metrics to calculate.\n
            Default: ``None``. This means that the default metrics will be used - ``Map``, ``NDCG``, ``Recall``.
        :param ks: highest k scores in ranking.\n
            Default: ``None``. This means that the default ``ks`` will be ``[1, 5, 10, 20]``.
        :param postprocessors: A list of postprocessors for modifying logits from the model.
            For example, it can be a softmax operation to logits or set the ``-inf`` value for some IDs.
            Default: ``None``.
        :param item_count: the total number of items in the dataset, required only for ``Coverage`` calculations.
            Default: ``None``.
        :param ground_truth_column: Name of key in batch that contains ground truth items.
        :param train_column: Name of key in batch that contains items on which the model is trained.
        """
        self._metrics = metrics
        self._ks = ks
        self._item_count = item_count
        self._metrics_builders: list[TorchMetricsBuilder] = []
        self._dataloaders_size: list[int] = []
        self._postprocessors: list[PostprocessorBase] = postprocessors or []
        self._ground_truth_column = ground_truth_column
        self._train_column = train_column

    def _get_dataloaders_size(self, dataloaders: Optional[Any]) -> list[int]:
        if isinstance(dataloaders, CombinedLoader):
            return [len(dataloader) for dataloader in dataloaders.flattened]  # pragma: no cover
        return [len(dataloaders)]

    def on_validation_epoch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        self._dataloaders_size = self._get_dataloaders_size(trainer.val_dataloaders)
        self._metrics_builders = [
            TorchMetricsBuilder(self._metrics, self._ks, self._item_count) for _ in self._dataloaders_size
        ]
        for builder in self._metrics_builders:
            builder.reset()

    def on_test_epoch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        self._dataloaders_size = self._get_dataloaders_size(trainer.test_dataloaders)
        self._metrics_builders = [
            TorchMetricsBuilder(self._metrics, self._ks, self._item_count) for _ in self._dataloaders_size
        ]
        for builder in self._metrics_builders:
            builder.reset()

    def _apply_postproccesors(self, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        for postprocessor in self._postprocessors:
            logits = postprocessor.on_validation(batch, logits)
        return logits

    def on_validation_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: LightningModule,
        outputs: InferenceOutput,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
        )

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: LightningModule,
        outputs: InferenceOutput,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:  # pragma: no cover
        self._batch_end(
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
        )

    def _batch_end(
        self,
        trainer: lightning.Trainer,  # noqa: ARG002
        pl_module: LightningModule,
        outputs: InferenceOutput,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        seen_scores = self._apply_postproccesors(batch, outputs["logits"])
        sampled_items = torch.topk(seen_scores, k=self._metrics_builders[dataloader_idx].max_k, dim=1).indices
        self._metrics_builders[dataloader_idx].add_prediction(
            sampled_items, batch[self._ground_truth_column], batch.get(self._train_column)
        )

        if batch_idx + 1 == self._dataloaders_size[dataloader_idx]:
            pl_module.log_dict(
                self._metrics_builders[dataloader_idx].get_metrics(),
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=True,
            )

    def on_validation_epoch_end(self, trainer: lightning.Trainer, pl_module: LightningModule) -> None:
        self._epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer: lightning.Trainer, pl_module: LightningModule) -> None:  # pragma: no cover
        self._epoch_end(trainer, pl_module)

    def _epoch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        @rank_zero_only
        def print_metrics() -> None:
            metrics = {}
            for name, value in trainer.logged_metrics.items():
                if "@" in name:
                    metrics[name] = value.item()

            if metrics:
                metrics_df = metrics_to_df(metrics)

                print(metrics_df)  # noqa: T201
                print()  # noqa: T201

        print_metrics()
