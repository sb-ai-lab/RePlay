from typing import Optional, Protocol

import lightning
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from replay.metrics.torch_metrics_builder import MetricName, TorchMetricsBuilder, metrics_to_df
from replay.models.nn.sequential.callbacks.validation_callback import ValidationBatch

from replay.models.nn.sequential.postprocessors import BasePostProcessor


UNSEEN_PREFIX_NAME = "unseen-"


class ValidationMetricsCallback(lightning.Callback):
    """
    Callback for validation and testing stages
    """

    def __init__(
        self,
        metrics: Optional[list[MetricName]] = None,
        ks: Optional[list[int]] = None,
        postprocessors: Optional[list[BasePostProcessor]] = None,
        item_count: Optional[int] = None,
    ):
        seen_metrics, unseen_metrics = self._separate_metrics(metrics)
        self._metrics_builder = TorchMetricsBuilder(seen_metrics, ks, item_count)
        self._unseen_metrics_builder = TorchMetricsBuilder(unseen_metrics, ks, item_count) if unseen_metrics else None
        self._postprocessors: list[BasePostProcessor] = postprocessors or []

    def _separate_metrics(
        self, metrics: Optional[list[MetricName]] = None
    ) -> tuple[Optional[list[MetricName]], Optional[list[MetricName]]]:
        if metrics is None:
            return None, None
        seen_metrics: list[MetricName] = []
        unseen_metrics: list[MetricName] = []
        for metric in metrics:
            if metric.startswith(UNSEEN_PREFIX_NAME):
                removed_unseen: MetricName = metric.replace(UNSEEN_PREFIX_NAME, "")
                unseen_metrics.append(removed_unseen)
            else:
                seen_metrics.append(metric)
        return (
            seen_metrics if len(seen_metrics) != 0 else None,
            unseen_metrics if len(unseen_metrics) != 0 else None,
        )

    def on_validation_epoch_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule  # noqa: ARG002
    ) -> None:
        self._metrics_builder.reset()
        if self._unseen_metrics_builder:
            self._unseen_metrics_builder.reset()

    def on_test_epoch_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule  # noqa: ARG002
    ) -> None:  # pragma: no cover
        self._metrics_builder.reset()
        if self._unseen_metrics_builder:
            self._unseen_metrics_builder.reset()

    def on_validation_batch_end(
        self,
        trainer: lightning.Trainer,  # noqa: ARG002
        pl_module: lightning.LightningModule,  # noqa: ARG002
        outputs: torch.Tensor,
        batch: ValidationBatch,
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        assert isinstance(outputs, torch.Tensor)

        self._batch_end(batch, outputs)

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,  # noqa: ARG002
        pl_module: lightning.LightningModule,  # noqa: ARG002
        outputs: torch.Tensor,
        batch: ValidationBatch,
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:  # pragma: no cover
        assert isinstance(outputs, torch.Tensor)
        self._batch_end(batch, outputs)

    def on_validation_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule  # noqa: ARG002
    ) -> None:
        self._epoch_end(pl_module)

    def on_test_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule  # noqa: ARG002
    ) -> None:  # pragma: no cover
        self._epoch_end(pl_module)

    def _compute_pipeline(
        self,
        batch: ValidationBatch,
        scores: torch.Tensor,
    ) -> torch.LongTensor:
        for postprocessor in self._postprocessors:
            scores = postprocessor.on_validation(batch, scores)
        return scores

    def _batch_end(self, batch: ValidationBatch, scores: torch.Tensor) -> None:
        seen_scores = self._compute_pipeline(batch, scores)
        sampled_items = torch.topk(seen_scores, k=self._metrics_builder.max_k, dim=1).indices
        self._metrics_builder.add_prediction(sampled_items, batch.ground_truth, batch.train)
        if self._unseen_metrics_builder is None:
            return
        if not hasattr(batch, "unseen_ground_truth"):
            error_msg = (
                "batch doesn't contain unseen_ground_truth attribute. "
                "Probable cause is ValidationDataset at validation_dataloader "
                "doesn't contain unseen_ground_truth SequentialDataset. "
                "Specify it, if not specified."
            )
            raise AttributeError(error_msg)
        unseen_scores = self._compute_pipeline(batch, scores)
        sampled_unseen_items = torch.topk(unseen_scores, k=self._metrics_builder.max_k, dim=1).indices
        self._unseen_metrics_builder.add_prediction(sampled_unseen_items, batch.unseen_ground_truth)

    def _epoch_end(self, pl_module: lightning.LightningModule) -> None:
        metrics: dict[str, float] = {}
        seen_metrics = self._metrics_builder.get_metrics()
        metrics = dict(seen_metrics)
        if self._unseen_metrics_builder:
            for key, value in self._unseen_metrics_builder.get_metrics().items():
                metrics[UNSEEN_PREFIX_NAME + key] = value
        pl_module.log_dict(metrics, on_epoch=True, sync_dist=True)

        @rank_zero_only
        def print_metrics() -> None:
            metrics = {}
            for value in pl_module._trainer._results.values():
                metrics[value.meta.name] = value.value.item()

            metrics_df = metrics_to_df(metrics)
            print(metrics_df)  # noqa: T201
            print()  # noqa: T201

        print_metrics()
