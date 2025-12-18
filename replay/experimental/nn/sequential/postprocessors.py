from typing import Literal, Optional, Protocol

import numpy as np
import pandas as pd
import torch

from replay.data.nn import TensorMap
from replay.models.nn.sequential.postprocessors._base import BasePostProcessor

__all__ = ["RemoveSeenItems"]


class CallbackBatch(Protocol):
    query_id: torch.Tensor
    features: TensorMap
    ground_truth: Optional[torch.Tensor]


class RemoveSeenItems(BasePostProcessor):
    """
    Filters out the items that already have been seen in dataset.

    Args:
        seen_path: Path to the parquet file containing users' interactions.
        item_count: Total number of unique items in the dataset (aka cardinality).
        query_column: Name of the column containing query ids. Default: ``"query_id"``
        item_column: Name of the column containing item ids. Default: ``"item_id"``
    """

    def __init__(
        self, seen_path: str, item_count: int, query_column: str = "query_id", item_column: str = "item_id"
    ) -> None:
        super().__init__()
        self.item_count = item_count
        self._candidates = None
        seen_data = pd.read_parquet(seen_path, columns=[query_column, item_column])

        max_seq_len = seen_data[item_column].str.len().max()
        seen_data[item_column] = seen_data[item_column].apply(
            lambda seq: np.concatenate(([seq[0]] * (max_seq_len - len(seq)), seq))
        )

        seen_data = seen_data.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        self.queries = torch.from_numpy(seen_data[query_column].to_numpy())
        self.sequences = torch.tensor(seen_data[item_column].tolist(), dtype=torch.long)

    def on_validation(
        self,
        batch: CallbackBatch,
        scores: torch.Tensor,
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        """
        Validation step.

        :param batch: Input batch pased to the callback.
        :param scores: calculated logits.

        :returns: modified query ids and scores and ground truth dataset
        """
        modified_scores = self._compute_scores(batch, scores.clone(), "val")
        return modified_scores

    def on_prediction(self, batch: CallbackBatch, scores: torch.Tensor) -> tuple[torch.LongTensor, torch.Tensor]:
        """
        Prediction step.

        :param batch: Input batch pased to the callback.
        :param scores: calculated logits.

        :returns: modified query ids and scores
        """
        modified_scores = self._compute_scores(batch, scores.clone(), "test")
        return modified_scores

    def _compute_scores(
        self, batch: CallbackBatch, scores: torch.Tensor, stage: Literal["val", "test"]
    ) -> torch.Tensor:
        self.queries = self.queries.to(batch.query_id.device)
        self.sequences = self.sequences.to(batch.query_id.device)

        query_mask = torch.isin(self.queries, batch.query_id)
        seen_ids = self.sequences[query_mask].to(device=scores.device)

        batch_factors = torch.arange(0, batch.query_id.numel(), device=scores.device) * self.item_count
        factored_ids = seen_ids + batch_factors.unsqueeze(1)
        seen_ids_flat = factored_ids.flatten()

        apply_candidates = stage == "test"
        if apply_candidates and self._candidates is not None:
            _scores = torch.full((scores.shape[0], self.item_count), -float("inf"), device=scores.device)
            _scores[:, self._candidates] = torch.reshape(scores, _scores[:, self._candidates].shape)
            scores = _scores
        if scores.is_contiguous():
            scores.view(-1)[seen_ids_flat] = -np.inf
        else:
            flat_scores = scores.flatten()
            flat_scores[seen_ids_flat] = -np.inf
            scores = flat_scores.reshape(scores.shape)
        return scores
