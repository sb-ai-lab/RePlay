from typing import Literal

import lightning
import torch

from replay.metrics.torch_metrics_builder import (
    DEFAULT_METRICS,
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
        metrics: list[MetricName] | None = None,
        ks: list[int] | None = None,
        postprocessors: list[PostprocessorBase] | None = None,
        item_count: int | None = None,
        ground_truth_column: str = "ground_truth",
        train_column: str = "train",
        verbose: bool = True,
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
        :param verbose: if ``True``, prints validation/test metrics to stdout after each epoch.
        """
        self._metrics = metrics or DEFAULT_METRICS
        self._ks = ks
        self._item_count = item_count
        self._metrics_builders: list[TorchMetricsBuilder] = []
        self._dataloaders_size: list[int] = []
        self._postprocessors: list[PostprocessorBase] = postprocessors or []
        self._ground_truth_column = ground_truth_column
        self._train_column = train_column
        self._verbose = verbose
        self._validation_metrics: dict[int, dict[str, float]] = {}
        self._test_metrics: dict[int, dict[str, float]] = {}

    def get_metrics(
        self,
        stage: Literal["validate", "test"] = "validate",
    ) -> dict[int, dict[str, float]]:
        """
        Returns metrics history by epoch for selected stage.

        The key is epoch index (0-based), and value is a dictionary with metric values.
        """
        metrics_by_stage = self._validation_metrics if stage == "validate" else self._test_metrics
        return {epoch: metrics.copy() for epoch, metrics in metrics_by_stage.items()}

    def state_dict(self) -> dict[str, dict[int, dict[str, float]]]:
        return {
            "validation_metrics": self._validation_metrics,
            "test_metrics": self._test_metrics,
        }

    def load_state_dict(self, state_dict: dict[str, dict[int, dict[str, float]]]) -> None:
        validation_metrics = state_dict.get("validation_metrics", {})
        self._validation_metrics = {
            int(epoch): {name: float(value) for name, value in metrics.items()}
            for epoch, metrics in validation_metrics.items()
        }
        test_metrics = state_dict.get("test_metrics", {})
        self._test_metrics = {
            int(epoch): {name: float(value) for name, value in metrics.items()}
            for epoch, metrics in test_metrics.items()
        }

    def on_validation_epoch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        self._epoch_start(dataloaders_size=trainer.num_val_batches)

    def on_test_epoch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        self._epoch_start(dataloaders_size=trainer.num_test_batches)

    def _epoch_start(self, dataloaders_size):
        self._dataloaders_size = dataloaders_size
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
    ) -> None:
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
        self._epoch_end(trainer, pl_module, is_validation=True)

    def on_test_epoch_end(self, trainer: lightning.Trainer, pl_module: LightningModule) -> None:
        self._epoch_end(trainer, pl_module, is_validation=False)

    def _epoch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        is_validation: bool,
    ) -> None:
        metrics = self._collect_logged_metrics(trainer)

        if is_validation:
            self._validation_metrics[int(trainer.current_epoch)] = metrics.copy()
        else:
            self._test_metrics[int(trainer.current_epoch)] = metrics.copy()

        if self._verbose:
            self._print_metrics(trainer, metrics)

    def _collect_logged_metrics(self, trainer: lightning.Trainer) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, value in trainer.logged_metrics.items():
            if "@" not in name or name.split("@")[0] not in self._metrics:
                continue
            if isinstance(value, torch.Tensor):
                metrics[name] = value.detach().cpu().item()
            else:
                metrics[name] = float(value)
        return metrics

    def _print_metrics(self, trainer: lightning.Trainer, metrics: dict[str, float]) -> None:
        if not trainer.is_global_zero:
            return
        if not metrics:
            return

        if len(self._dataloaders_size) > 1:
            for i in range(len(self._dataloaders_size)):
                suffix = trainer._results.DATALOADER_SUFFIX.format(i)[1:]
                cur_dataloader_metrics = {k.split("/")[0]: v for k, v in metrics.items() if suffix in k}
                if not cur_dataloader_metrics:
                    continue
                metrics_df = metrics_to_df(cur_dataloader_metrics)

                print(suffix)  # noqa: T201
                print(metrics_df, "\n")  # noqa: T201
        else:
            metrics_df = metrics_to_df(metrics)
            print(metrics_df, "\n")  # noqa: T201
