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
    A callback for validation and testing stages.

    If multiple validation/testing dataloaders are used,
    the suffix of the metric name will contain the serial number of the dataloader.

    For the correct calculation of metrics inside the callback,
    the batch must contain the ``ground_truth_column`` key - the padding value of this tensor can be any,
    the main condition is that the padding value does not overlap with the existing item ID values.
    For example, these can be negative values.

    To calculate the ``coverage`` and ``novelty`` metrics, the batch must additionally contain the ``train_column`` key.
    The padding value of this tensor can be any, the main condition is that the padding value does not overlap
    with the existing item ID values. For example, these can be negative values.

    When only selected candidates are scored, their global item IDs must be supplied through
    ``LightningModule.candidates_to_score`` or the batch key ``candidates_to_score``. A one-dimensional
    tensor is shared by the batch; a two-dimensional batch tensor provides candidates per row.
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
        :param metrics: A sequence of metrics to calculate.\n
            Default: ``None``. This means that the default metrics will be used - ``Map``, ``NDCG``, ``Recall``.
        :param ks: the highest k scores in ranking.\n
            Default: ``None``. This means that the default ``ks`` will be ``[1, 5, 10, 20]``.
        :param postprocessors: A list of postprocessors for modifying logits from the model.
            For example, it can be a softmax operation to logits or set the ``-inf`` value for some IDs.
            Default: ``None``.
        :param item_count: the total number of items in the dataset, required only for ``Coverage`` calculations.
            Default: ``None``.
        :param ground_truth_column: A name of the key in a batch that contains ground truth items.
            Default: ``"ground_truth"``.
        :param train_column: A name of the key in a batch that contains items on which the model is trained.
            Default: ``"train"``.
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
        self._candidates: torch.LongTensor | None = None
        self._device_candidates: torch.LongTensor | None = None

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
        pl_module: LightningModule,
    ) -> None:
        self._set_candidates(pl_module)
        self._epoch_start(dataloaders_size=trainer.num_val_batches)

    def on_test_epoch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._set_candidates(pl_module)
        self._epoch_start(dataloaders_size=trainer.num_test_batches)

    def _set_candidates(self, pl_module: LightningModule) -> None:
        candidates = getattr(pl_module, "candidates_to_score", None)
        if candidates is not None:
            if not isinstance(candidates, torch.Tensor):
                msg = "Candidates must be a tensor or None."
                raise TypeError(msg)
            self._validate_candidates(candidates, allow_row_wise=False)
        self._candidates = candidates
        self._device_candidates = None
        self._set_postprocessor_candidates(candidates)

    def _validate_candidates(self, candidates: torch.Tensor, *, allow_row_wise: bool) -> None:
        valid_dimensions = (1, 2) if allow_row_wise else (1,)
        if candidates.ndim not in valid_dimensions:
            msg = "Candidates must be a one-dimensional tensor."
            if allow_row_wise:
                msg = "Batch candidates must be a one- or two-dimensional tensor."
            raise ValueError(msg)
        if candidates.dtype != torch.long:
            msg = "Candidates must have torch.long dtype."
            raise TypeError(msg)
        if candidates.shape[-1] == 0:
            msg = "Candidates must be non-empty."
            raise ValueError(msg)
        if candidates.ndim == 1:
            has_duplicates = torch.unique(candidates).numel() != candidates.numel()
        else:
            sorted_candidates = torch.sort(candidates, dim=1).values
            has_duplicates = (sorted_candidates[:, 1:] == sorted_candidates[:, :-1]).any()
        if has_duplicates:
            msg = "The tensor of candidates to score must be unique."
            raise ValueError(msg)
        if (candidates < 0).any():
            msg = "Candidate IDs must be non-negative."
            raise ValueError(msg)
        if self._item_count is not None and (candidates >= self._item_count).any():
            msg = "Candidate IDs must be less than item_count."
            raise ValueError(msg)

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

    def _prepare_candidates(self, batch: dict, scores: torch.Tensor) -> torch.LongTensor | None:
        if "candidates_to_score" in batch and batch["candidates_to_score"] is not self._candidates:
            candidates = batch["candidates_to_score"]
            if candidates is None:
                self._set_postprocessor_candidates(None)
                return None
            if not isinstance(candidates, torch.Tensor):
                msg = "Batch candidates must be a tensor or None."
                raise TypeError(msg)
            self._validate_candidates(candidates, allow_row_wise=True)
            if candidates.ndim == 2 and candidates.shape[0] != scores.shape[0]:
                msg = "Row-wise candidates and logits must have the same batch size."
                raise ValueError(msg)
            candidates = candidates.to(scores.device)
        elif self._candidates is None:
            self._set_postprocessor_candidates(None)
            return None
        elif self._candidates.device == scores.device:
            candidates = self._candidates
        else:
            if self._device_candidates is None or self._device_candidates.device != scores.device:
                self._device_candidates = self._candidates.to(scores.device)
            candidates = self._device_candidates

        if candidates.shape[-1] != scores.shape[1]:
            msg = "The number of candidates must match the logits width."
            raise ValueError(msg)
        self._set_postprocessor_candidates(candidates)
        return candidates

    def _set_postprocessor_candidates(self, candidates: torch.LongTensor | None) -> None:
        for postprocessor in self._postprocessors:
            if (
                candidates is not None
                and candidates.ndim == 2
                and not getattr(postprocessor, "_supports_row_wise_candidates", False)
            ):
                msg = "Row-wise candidates require a postprocessor that supports per-row candidate catalogs."
                raise ValueError(msg)
            if getattr(postprocessor, "candidates", None) is not candidates:
                postprocessor.candidates = candidates

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
        trainer: lightning.Trainer,
        pl_module: LightningModule,
        outputs: InferenceOutput,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        candidates = self._prepare_candidates(batch, outputs["logits"])
        seen_scores = self._apply_postproccesors(batch, outputs["logits"])
        max_k = self._metrics_builders[dataloader_idx].max_k
        if max_k > seen_scores.shape[1]:
            msg = "The largest k must not exceed the number of score columns."
            raise ValueError(msg)
        sampled_items = torch.topk(seen_scores, k=max_k, dim=1).indices
        if candidates is not None:
            if candidates.ndim == 1:
                sampled_items = candidates[sampled_items]
            else:
                sampled_items = torch.gather(candidates, 1, sampled_items)
        self._metrics_builders[dataloader_idx].add_prediction(
            sampled_items, batch[self._ground_truth_column], batch.get(self._train_column)
        )
        if batch_idx + 1 == self._dataloaders_size[dataloader_idx] and not trainer.sanity_checking:
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
        if trainer.sanity_checking:
            return
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
            metrics[name] = value.detach().cpu().item()
        return metrics

    def _print_metrics(self, trainer: lightning.Trainer, metrics: dict[str, float]) -> None:
        if not trainer.is_global_zero:  # pragma: no cover
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
