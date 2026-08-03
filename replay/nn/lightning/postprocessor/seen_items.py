import torch

from replay.data.nn import TensorMap

from ._base import PostprocessorBase

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class SeenItemsFilter(PostprocessorBase):
    """
    Masks (sets logits value to ``-inf``) the items that already have been seen in the given dataset
    (i.e. in the sequence of items for that logits are calculated).\n
    Should be used in Lightning callbacks for inferencing or metrics computing.

    .. rubric:: Input example

    logits [B=2 users, I=3 items]::

        logits =
        [[0.1, 0.2, 0.3],    # user0
        [-0.1, -0.2, -0.3]]  # user1

    Seen items per user::

        seen_items =
        user0: [1, 0]
        user1: [1, 2, 1]

    .. rubric:: Output example

    SeenItemsFilter sets logits of seen items to ``-inf``::

        processed_logits =
        [[   -inf,    -inf,  0.3000], # user0
        [-0.1000,    -inf,    -inf]]  # user1
    """

    _supports_row_wise_candidates = True

    def __init__(self, item_count: int, seen_items_column: str = "seen_ids") -> None:
        """
        :param item_count: A total number of items that the model knows about (``cardinality``).
            It is recommended to take this value from ``TensorSchema``. \n
            Please note that values outside the range [0, `item_count-1`] are filtered out (considered as padding).
        :param seen_items_column: A name of the column in a batch that contains users' interactions (seen item IDs).
        """
        if item_count <= 0:
            msg = "item_count must be positive."
            raise ValueError(msg)
        super().__init__()
        self.item_count = item_count
        self.seen_items_column = seen_items_column
        self._candidates: torch.LongTensor | None = None
        self._candidate_lookup: torch.LongTensor | None = None
        self._candidate_lookup_device: torch.device | None = None

    @property
    def candidates(self) -> torch.LongTensor | None:
        """Return global item IDs represented by score columns."""
        return self._candidates

    @candidates.setter
    def candidates(self, candidates: torch.LongTensor | None = None) -> None:
        """Set global item IDs represented by score columns."""
        if candidates is not None:
            self._validate_candidates(candidates)
        self._candidates = candidates
        self._candidate_lookup = None
        self._candidate_lookup_device = None

    def on_validation(self, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        return self._compute_scores(batch, logits.detach().clone())

    def on_prediction(self, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        return self._compute_scores(batch, logits.detach().clone())

    def _compute_scores(self, batch: TensorMap, logits: torch.Tensor) -> torch.Tensor:
        seen_ids = self._validate_inputs(batch, logits).to(device=logits.device, dtype=torch.long)
        valid_seen = (seen_ids >= 0) & (seen_ids < self.item_count)

        if self._candidates is None:
            local_columns = seen_ids
        elif self._candidates.ndim == 1:
            lookup = self._get_candidate_lookup(logits.device)
            local_columns = lookup[seen_ids.clamp(min=0, max=self.item_count - 1)]
            valid_seen &= local_columns >= 0
        else:
            candidates = self._candidates.to(logits.device)
            sorted_candidates, local_columns_by_rank = torch.sort(candidates, dim=1)
            safe_seen = seen_ids.clamp(min=0, max=self.item_count - 1)
            insertion_points = torch.searchsorted(sorted_candidates, safe_seen.contiguous())
            bounded_points = insertion_points.clamp(max=candidates.shape[1] - 1)
            valid_seen &= insertion_points < candidates.shape[1]
            valid_seen &= torch.gather(sorted_candidates, 1, bounded_points) == safe_seen
            local_columns = torch.gather(local_columns_by_rank, 1, bounded_points)

        rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1).expand_as(seen_ids)
        logits[rows[valid_seen], local_columns[valid_seen]] = -torch.inf
        return logits

    def _validate_inputs(self, batch: TensorMap, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            msg = "logits must be a rank-two tensor."
            raise ValueError(msg)
        if not logits.is_floating_point():
            msg = "logits must have a floating-point dtype."
            raise TypeError(msg)

        seen_ids = batch[self.seen_items_column]
        if not isinstance(seen_ids, torch.Tensor):
            msg = f"batch[{self.seen_items_column!r}] must be a tensor."
            raise TypeError(msg)
        if seen_ids.dtype not in _INTEGER_DTYPES:
            msg = f"batch[{self.seen_items_column!r}] must have an integer dtype."
            raise TypeError(msg)
        if seen_ids.ndim != 2:
            msg = f"batch[{self.seen_items_column!r}] must be a rank-two tensor."
            raise ValueError(msg)
        if seen_ids.shape[0] != logits.shape[0]:
            msg = "Seen items and logits must have the same batch size."
            raise ValueError(msg)
        if self._candidates is not None and self._candidates.ndim == 2 and self._candidates.shape[0] != logits.shape[0]:
            msg = "Row-wise candidates and logits must have the same batch size."
            raise ValueError(msg)
        expected_width = self.item_count if self._candidates is None else self._candidates.shape[-1]
        if logits.shape[1] != expected_width:
            msg = "The logits width must match item_count or the number of candidates."
            raise ValueError(msg)
        return seen_ids

    def _validate_candidates(self, candidates: torch.Tensor) -> None:
        if candidates.ndim not in (1, 2):
            msg = "Candidates must be a one- or two-dimensional tensor."
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
        if ((candidates < 0) | (candidates >= self.item_count)).any():
            msg = "Candidate IDs must be in the range [0, item_count)."
            raise ValueError(msg)

    def _get_candidate_lookup(self, device: torch.device) -> torch.LongTensor:
        candidates = self._candidates
        if candidates is None or candidates.ndim != 1:  # pragma: no cover
            msg = "A shared candidate vector is required for a catalog lookup."
            raise RuntimeError(msg)
        if self._candidate_lookup_device != device:
            candidates = candidates.to(device)
            lookup = torch.full((self.item_count,), -1, dtype=torch.long, device=device)
            lookup[candidates] = torch.arange(candidates.numel(), dtype=torch.long, device=device)
            self._candidate_lookup = lookup
            self._candidate_lookup_device = device
        return self._candidate_lookup
