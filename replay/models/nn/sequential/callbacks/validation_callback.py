from typing import Any, List, Optional, Protocol, Tuple, Literal

import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from replay.metrics.torch_metrics_builder import TorchMetricsBuilder, metrics_to_df
from replay.models.nn.sequential.postprocessors import BasePostProcessor


CallbackMetricName = Literal[
    "recall",
    "precision",
    "ndcg",
    "map",
    "mrr",
    "novelty",
    "coverage",
]


# pylint: disable=too-few-public-methods
class ValidationBatch(Protocol):
    """
    Validation callback batch
    """
    query_id: torch.LongTensor
    ground_truth: torch.LongTensor
    train: torch.LongTensor


class ValidationMetricsCallback(L.Callback):
    """
    Callback for validation and testing stages.

    If multiple validation/testing dataloaders are used,
    the suffix of the metric name will contain the serial number of the dataloader.
    """

    # pylint: disable=invalid-name
    def __init__(
        self,
        metrics: Optional[List[CallbackMetricName]] = None,
        ks: Optional[List[int]] = None,
        postprocessors: Optional[List[BasePostProcessor]] = None,
        item_count: Optional[int] = None,
    ):
        """
        :param metrics: Sequence of metrics to calculate.
        :param ks: highest k scores in ranking. Default: will be `[1, 5, 10, 20]`.
        :param postprocessors: postprocessors to validation stage.
        :param item_count: the total number of items in the dataset, required only for Coverage calculations.
        """
        self._metrics = metrics
        self._ks = ks
        self._item_count = item_count
        self._metrics_builders: List[TorchMetricsBuilder] = []
        self._dataloaders_size: List[int] = []
        self._postprocessors: List[BasePostProcessor] = postprocessors or []

    def _get_dataloaders_size(self, dataloaders: Optional[Any]) -> List[int]:
        if isinstance(dataloaders, torch.utils.data.DataLoader):
            return [len(dataloaders)]
        return [len(dataloader) for dataloader in dataloaders]

    # pylint: disable=unused-argument
    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._dataloaders_size = self._get_dataloaders_size(trainer.val_dataloaders)
        self._metrics_builders = [
            TorchMetricsBuilder(self._metrics, self._ks, self._item_count) for _ in self._dataloaders_size
        ]
        for builder in self._metrics_builders:
            builder.reset()

    # pylint: disable=unused-argument
    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # pragma: no cover
        self._dataloaders_size = self._get_dataloaders_size(trainer.test_dataloaders)
        self._metrics_builders = [
            TorchMetricsBuilder(self._metrics, self._ks, self._item_count) for _ in self._dataloaders_size
        ]
        for builder in self._metrics_builders:
            builder.reset()

    def _compute_pipeline(
        self, query_ids: torch.LongTensor, scores: torch.Tensor, ground_truth: torch.LongTensor
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        for postprocessor in self._postprocessors:
            query_ids, scores, ground_truth = postprocessor.on_validation(query_ids, scores, ground_truth)
        return query_ids, scores, ground_truth

    # pylint: disable=too-many-arguments
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor,
        batch: ValidationBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    # pylint: disable=unused-argument, too-many-arguments
    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor,
        batch: ValidationBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:  # pragma: no cover
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    # pylint: disable=too-many-arguments
    def _batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor,
        batch: ValidationBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        _, seen_scores, seen_ground_truth = self._compute_pipeline(batch.query_id, outputs, batch.ground_truth)
        sampled_items = torch.topk(seen_scores, k=self._metrics_builders[dataloader_idx].max_k, dim=1).indices
        self._metrics_builders[dataloader_idx].add_prediction(sampled_items, seen_ground_truth, batch.train)

        if batch_idx + 1 == self._dataloaders_size[dataloader_idx]:
            pl_module.log_dict(
                self._metrics_builders[dataloader_idx].get_metrics(),
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=True
            )

    # pylint: disable=unused-argument
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._epoch_end(trainer, pl_module)

    # pylint: disable=unused-argument
    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # pragma: no cover
        self._epoch_end(trainer, pl_module)

    # pylint: disable=unused-argument
    def _epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # pylint: disable=W0212
        @rank_zero_only
        def print_metrics() -> None:
            metrics = {}
            for name, value in trainer.logged_metrics.items():
                if '@' in name:
                    metrics[name] = value.item()

            if metrics:
                metrics_df = metrics_to_df(metrics)

                print(metrics_df)
                print()

        print_metrics()
