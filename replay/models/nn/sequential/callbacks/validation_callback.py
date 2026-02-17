from typing import Any, Literal, Protocol

import lightning
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing_extensions import deprecated

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


@deprecated("`ValidationBatch` class is deprecated.", stacklevel=2)
class ValidationBatch(Protocol):
    """
    Validation callback batch
    """

    query_id: torch.LongTensor
    ground_truth: torch.LongTensor
    train: torch.LongTensor


@deprecated(
    "`ValidationMetricsCallback` class is deprecated. "
    "Use `replay.nn.lightning.callback.ComputeMetricsCallback` instead."
)
class ValidationMetricsCallback(lightning.Callback):
    """
    Callback for validation and testing stages.

    If multiple validation/testing dataloaders are used,
    the suffix of the metric name will contain the serial number of the dataloader.

    For the callback to work correctly, the batch must contain the `query_id` and `ground_truth` keys.
    If you want to calculate the coverage or novelty metrics then the batch must additionally contain the `train` key.
    """

    def __init__(
        self,
        metrics: list[CallbackMetricName] | None = None,
        ks: list[int] | None = None,
        postprocessors: list[BasePostProcessor] | None = None,
        item_count: int | None = None,
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
        self._metrics_builders: list[TorchMetricsBuilder] = []
        self._dataloaders_size: list[int] = []
        self._postprocessors: list[BasePostProcessor] = postprocessors or []

    def _get_dataloaders_size(self, dataloaders: Any | None) -> list[int]:
        if isinstance(dataloaders, torch.utils.data.DataLoader):
            return [len(dataloaders)]
        return [len(dataloader) for dataloader in dataloaders]

    def on_validation_epoch_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule  # noqa: ARG002
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
        pl_module: lightning.LightningModule,  # noqa: ARG002
    ) -> None:  # pragma: no cover
        self._dataloaders_size = self._get_dataloaders_size(trainer.test_dataloaders)
        self._metrics_builders = [
            TorchMetricsBuilder(self._metrics, self._ks, self._item_count) for _ in self._dataloaders_size
        ]
        for builder in self._metrics_builders:
            builder.reset()

    def _compute_pipeline(
        self, query_ids: torch.LongTensor, scores: torch.Tensor, ground_truth: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        for postprocessor in self._postprocessors:
            query_ids, scores, ground_truth = postprocessor.on_validation(query_ids, scores, ground_truth)
        return query_ids, scores, ground_truth

    def on_validation_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: torch.Tensor,
        batch: ValidationBatch | dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: torch.Tensor,
        batch: ValidationBatch | dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:  # pragma: no cover
        self._batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def _batch_end(
        self,
        trainer: lightning.Trainer,  # noqa: ARG002
        pl_module: lightning.LightningModule,
        outputs: torch.Tensor,
        batch: ValidationBatch | dict,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        _, seen_scores, seen_ground_truth = self._compute_pipeline(
            batch["query_id"] if isinstance(batch, dict) else batch.query_id,
            outputs,
            batch["ground_truth"] if isinstance(batch, dict) else batch.ground_truth,
        )
        sampled_items = torch.topk(seen_scores, k=self._metrics_builders[dataloader_idx].max_k, dim=1).indices
        self._metrics_builders[dataloader_idx].add_prediction(
            sampled_items,
            seen_ground_truth,
            batch.get("train") if isinstance(batch, dict) else batch.train,
        )

        if batch_idx + 1 == self._dataloaders_size[dataloader_idx]:
            pl_module.log_dict(
                self._metrics_builders[dataloader_idx].get_metrics(),
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=True,
            )

    def on_validation_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        self._epoch_end(trainer, pl_module)

    def on_test_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:  # pragma: no cover
        self._epoch_end(trainer, pl_module)

    def _epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:  # noqa: ARG002
        @rank_zero_only
        def print_metrics() -> None:
            metrics = {}

            for name, value in trainer.logged_metrics.items():
                if "@" in name:
                    metrics[name] = value.item()

            if not metrics:
                return

            if len(self._dataloaders_size) > 1:
                for i in range(len(self._dataloaders_size)):
                    suffix = trainer._results.DATALOADER_SUFFIX.format(i)[1:]
                    cur_dataloader_metrics = {k.split("/")[0]: v for k, v in metrics.items() if suffix in k}
                    metrics_df = metrics_to_df(cur_dataloader_metrics)

                    print(suffix)  # noqa: T201
                    print(metrics_df, "\n")  # noqa: T201
            else:
                metrics_df = metrics_to_df(metrics)
                print(metrics_df, "\n")  # noqa: T201

        print_metrics()
