import torch

from replay.data.nn import TensorMap

from ._base import PostprocessorBase


class SeenItemsFilter(PostprocessorBase):
    """
    Masks (sets logits value to ``-inf``) the items that already have been seen in the given dataset
    (i.e. in the sequence of items for that logits are calculated).\n
    Should be used in Lightning callbacks for inferencing or metrics computing.

    Example:
    --------

    Input logits [B=2 users, I=3 items]::

        logits =
        [[0.1, 0.2, 0.3],    # user0
         [-0.1, -0.2, -0.3]] # user1

    Seen items per user::

        seen_items =
            user0: [1, 0]
            user1: [1, 2, 1]

    SeenItemsFilter sets logits of seen items to ``-inf``::
        processed_logits =
        [[   -inf,    -inf,  0.3000], # user0
         [-0.1000,    -inf,    -inf]] # user1

    """

    def __init__(self, item_count: int, seen_items_column="seen_ids") -> None:
        """
        :param item_count: Total number of items that the model knows about (``cardinality``).
            It is recommended to take this value from ``TensorSchema``. \n
            Please note that values ​​outside the range [0, `item_count-1`] are filtered out (considered as padding).
        :param seen_items_column: Name of the column in batch that contains users' interactions (seen item ids).
        """
        super().__init__()
        self.item_count = item_count
        self.seen_items_column = seen_items_column
        self._candidates = None

    def on_validation(self, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        return self._compute_scores(batch, logits.detach().clone(), True)

    def on_prediction(self, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        return self._compute_scores(batch, logits.detach().clone(), False)

    def _compute_scores(
        self,
        batch: TensorMap,
        logits: torch.Tensor,
        is_validation: bool,
    ) -> torch.Tensor:
        seen_ids_padded = batch[self.seen_items_column]
        padding_mask = (seen_ids_padded < self.item_count) & (seen_ids_padded >= 0)

        batch_factors = torch.arange(0, batch["query_id"].numel()) * self.item_count
        factored_ids = seen_ids_padded + batch_factors.unsqueeze(1)
        seen_ids_flat = factored_ids[padding_mask]

        if not is_validation and self._candidates is not None:
            _logits = torch.full((logits.size(0), self.item_count), -torch.inf)
            _logits[:, self._candidates] = torch.reshape(logits, _logits[:, self.candidates].shape)
            logits = _logits

        if logits.is_contiguous():
            logits.view(-1)[seen_ids_flat] = -torch.inf
        else:
            flat_scores = logits.flatten()
            flat_scores[seen_ids_flat] = -torch.inf
            logits = flat_scores.reshape(logits.shape)

        if not is_validation and self._candidates is not None:
            logits = logits[:, self._candidates]

        return logits
