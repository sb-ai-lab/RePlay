from typing import Dict, List, Optional, Protocol, Tuple, Literal

import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from replay.metrics.torch_metrics_builder import TorchMetricsBuilder, metrics_to_df
from replay.models.nn.sequential.postprocessors import BasePostProcessor


CallbackMetricName = Literal[
    "recall",
    "unseen-recall",
    "precision",
    "unseen-precision",
    "ndcg",
    "unseen-ndcg",
    "map",
    "unseen-map",
    "mrr",
    "unseen-mrr",
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
    Callback for validation and testing stages
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
        :param item_count: the total number of items in the dataset.
        """
        self._metrics_builder = TorchMetricsBuilder(metrics, ks, item_count)
        self._postprocessors: List[BasePostProcessor] = postprocessors or []

    # pylint: disable=unused-argument
    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._metrics_builder.reset()

    # pylint: disable=unused-argument
    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # pragma: no cover
        self._metrics_builder.reset()

    # pylint: disable=unused-argument, too-many-arguments
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor,
        batch: ValidationBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert isinstance(outputs, torch.Tensor)
        self._batch_end(batch.query_id, outputs, batch.ground_truth, batch.train)

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
        assert isinstance(outputs, torch.Tensor)
        self._batch_end(batch.query_id, outputs, batch.ground_truth, batch.train)

    # pylint: disable=unused-argument
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._epoch_end(pl_module)

    # pylint: disable=unused-argument
    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # pragma: no cover
        self._epoch_end(pl_module)

    def _compute_pipeline(
        self, query_ids: torch.LongTensor, scores: torch.Tensor, ground_truth: torch.LongTensor
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        for postprocessor in self._postprocessors:
            query_ids, scores, ground_truth = postprocessor.on_validation(query_ids, scores, ground_truth)
        return query_ids, scores, ground_truth

    def _batch_end(
        self, query_ids: torch.LongTensor, scores: torch.Tensor, ground_truth: torch.LongTensor, train: torch.LongTensor
    ) -> None:
        _, seen_scores, seen_ground_truth = self._compute_pipeline(query_ids, scores, ground_truth)
        sampled_items = torch.topk(seen_scores, k=self._metrics_builder.max_k, dim=1).indices
        self._metrics_builder.add_prediction(sampled_items, seen_ground_truth, train)

    def _epoch_end(self, pl_module: L.LightningModule) -> None:
        metrics: Dict[str, float] = {}
        seen_metrics = self._metrics_builder.get_metrics()
        for key, value in seen_metrics.items():
            metrics[key] = value
        pl_module.log_dict(metrics, on_epoch=True, sync_dist=True)

        # pylint: disable=W0212
        @rank_zero_only
        def print_metrics() -> None:
            metrics = {}
            for _, value in pl_module._trainer._results.items():
                metrics[value.meta.name] = value.value.item()

            metrics_df = metrics_to_df(metrics)
            print(metrics_df)
            print()

        print_metrics()
