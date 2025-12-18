import numpy as np
import pandas as pd
import torch

from replay.data.nn import TensorMap

from ._base import PostprocessorBase


class SeenItemsFilter(PostprocessorBase):
    """
    Filters out (sets logits value to ``-inf``) the items that already have been seen in given dataset.
    """

    def __init__(
        self,
        seen_path: str,
        item_count: int,
        query_column: str = "query_id",
        item_column: str = "item_id",
    ) -> None:
        """
        :param seen_path: Path to the parquet file containing users' interactions.
        :param item_count: Total number of items that the model knows about (aka ``cardinality``).
            Not in all cases, this value can be taken as the number of unique elements in the column
            of the dataset being submitted. It is recommended to take this value from ``TensorSchema``.
        :param query_column: Name of the column containing query ids. Default: ``"query_id"``.
        :param item_column: Name of the column containing item ids. Default: ``"item_id"``.
        """
        super().__init__()
        self.item_count = item_count
        self.candidates = None
        seen_data = pd.read_parquet(seen_path, columns=[query_column, item_column])

        max_seq_len = seen_data[item_column].str.len().max()
        seen_data[item_column] = seen_data[item_column].apply(
            lambda seq: np.concatenate(([seq[0]] * (max_seq_len - len(seq)), seq))
        )

        seen_data = seen_data.map(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

        self.queries = torch.from_numpy(seen_data[query_column].to_numpy())
        self.sequences = torch.tensor(seen_data[item_column].tolist(), dtype=torch.long)

    def on_validation(self, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        return self._compute_scores(batch, logits, True)

    def on_prediction(self, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        return self._compute_scores(batch, logits, False)

    def _compute_scores(
        self,
        batch: TensorMap,
        logits: torch.Tensor,
        is_validation: bool,
    ) -> torch.Tensor:
        device = batch["query_id"].device
        self.queries = self.queries.to(device)

        # in order to save GPU memory,
        # only those sequences that intersect
        # with the batch in the ``query_column`` are moved to the device.
        query_mask = torch.isin(self.queries, batch["query_id"]).cpu()
        seen_ids = self.sequences[query_mask].to(device=device)

        batch_factors = (
            torch.arange(0, batch["query_id"].numel(), device=device) * self.item_count
        )
        factored_ids = seen_ids + batch_factors.unsqueeze(1)
        seen_ids_flat = factored_ids.flatten()

        if not is_validation and self.candidates is not None:
            _logits = torch.full(
                (logits.size(0), self.item_count), -float("inf"), device=device
            )
            _logits[:, self.candidates] = torch.reshape(
                logits, _logits[:, self.candidates].shape
            )
            logits = _logits
        if logits.is_contiguous():
            logits.view(-1)[seen_ids_flat] = -np.inf
        else:
            flat_scores = logits.flatten()
            flat_scores[seen_ids_flat] = -np.inf
            logits = flat_scores.reshape(logits.shape)
        return logits
