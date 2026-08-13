import torch

from replay.data.nn import TensorMap
from replay.nn.loss.base import mask_negative_logits
from replay.nn.loss.ce import CESampled


class GroupedCESampled(CESampled):
    """Sampled CE with one independent negative pool per logical batch group.

    Negative pools must contain unique item IDs. Group losses are divided by
    ``groups_per_batch`` even for the final short batch. This preserves the
    weighting of the original smaller batches when the physical batch size is
    increased together with gradient accumulation.
    """

    def __init__(
        self,
        logical_batch_size: int,
        groups_per_batch: int,
        expected_num_negatives: int | None = None,
        cardinality: int | None = None,
        negative_labels_ignore_index: int = -100,
        **kwargs,
    ) -> None:
        """
        :param logical_batch_size: Number of rows that share one negative pool.
        :param groups_per_batch: Number of logical groups in a full physical batch.
        :param expected_num_negatives: Optional required size of every negative pool.
        :param cardinality: Optional catalog size enabling collision masking without a
            target-by-negative comparison matrix.
        :param negative_labels_ignore_index: Value ignored in negative labels.
        :param kwargs: Arguments passed to :class:`torch.nn.CrossEntropyLoss`.
        """
        if logical_batch_size <= 0:
            msg = "The logical_batch_size parameter must be positive."
            raise ValueError(msg)
        if groups_per_batch <= 1:
            msg = "The groups_per_batch parameter must be greater than one."
            raise ValueError(msg)
        if expected_num_negatives is not None and expected_num_negatives <= 0:
            msg = "The expected_num_negatives parameter must be positive."
            raise ValueError(msg)
        if cardinality is not None and cardinality <= 0:
            msg = "The cardinality parameter must be positive."
            raise ValueError(msg)
        super().__init__(
            negative_labels_ignore_index=negative_labels_ignore_index, **kwargs
        )
        if self._loss.reduction != "mean":
            msg = "GroupedCESampled supports only mean reduction."
            raise ValueError(msg)
        self.logical_batch_size = logical_batch_size
        self.groups_per_batch = groups_per_batch
        self.expected_num_negatives = expected_num_negatives
        self.cardinality = cardinality
        self.register_buffer("_negative_column_lookup", None, persistent=False)

    def _validate_inputs(
        self,
        model_embeddings: torch.Tensor,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> int:
        if model_embeddings.dim() != 3:
            msg = "model_embeddings must have shape [batch, sequence, embedding]."
            raise ValueError(msg)
        if positive_labels.dim() != 3 or positive_labels.size(-1) != 1:
            msg = "GroupedCESampled requires exactly one positive label per sequence position."
            raise ValueError(msg)
        if positive_labels.shape[:2] != model_embeddings.shape[:2]:
            msg = "positive_labels and model_embeddings must have equal batch and sequence dimensions."
            raise ValueError(msg)
        if target_padding_mask.shape != positive_labels.shape:
            msg = "target_padding_mask must have the same shape as positive_labels."
            raise ValueError(msg)
        if (
            negative_labels.dim() != 2
            or negative_labels.size(0) != self.groups_per_batch
        ):
            msg = f"negative_labels must have shape [{self.groups_per_batch}, num_negatives]."
            raise ValueError(msg)
        if (
            self.expected_num_negatives is not None
            and negative_labels.size(1) != self.expected_num_negatives
        ):
            msg = f"Each negative pool must contain {self.expected_num_negatives} items, got {negative_labels.size(1)}."
            raise ValueError(msg)
        if model_embeddings.size(0) == 0:
            msg = "GroupedCESampled does not support empty batches."
            raise ValueError(msg)

        active_groups = (
            model_embeddings.size(0) + self.logical_batch_size - 1
        ) // self.logical_batch_size
        if active_groups > self.groups_per_batch:
            capacity = self.logical_batch_size * self.groups_per_batch
            msg = f"Batch size {model_embeddings.size(0)} exceeds grouped loss capacity {capacity}."
            raise ValueError(msg)
        return active_groups

    def _loss_from_logits(
        self,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
    ) -> torch.Tensor:
        negative_logits = self._mask_negative_logits(
            negative_logits, negative_labels, positive_labels
        )
        logits = torch.cat((positive_logits, negative_logits), dim=-1)
        target = torch.zeros(
            positive_logits.size(0), dtype=torch.long, device=logits.device
        )
        return self._loss(logits, target)

    def _mask_negative_logits(
        self,
        negative_logits: torch.Tensor,
        negative_labels: torch.LongTensor,
        positive_labels: torch.LongTensor,
    ) -> torch.Tensor:
        if self.cardinality is None:
            return mask_negative_logits(
                negative_logits,
                negative_labels,
                positive_labels.unsqueeze(-1),
                self.negative_labels_ignore_index,
            )

        lookup = self._negative_column_lookup
        if lookup is None or lookup.device != negative_labels.device:
            lookup = torch.empty(
                self.cardinality, dtype=torch.int32, device=negative_labels.device
            )
            self._negative_column_lookup = lookup
        lookup.fill_(-1)

        valid_negatives = negative_labels.ge(0) & negative_labels.lt(self.cardinality)
        negative_columns = torch.arange(
            negative_labels.numel(), dtype=lookup.dtype, device=negative_labels.device
        )
        lookup[negative_labels[valid_negatives]] = negative_columns[valid_negatives]

        valid_positives = positive_labels.ge(0) & positive_labels.lt(self.cardinality)
        safe_positives = positive_labels.clamp(min=0, max=self.cardinality - 1)
        collision_columns = lookup[safe_positives].long()
        collision_rows = torch.arange(
            positive_labels.numel(), device=positive_labels.device
        )
        collisions = valid_positives & collision_columns.ge(0)
        masked_value = max(-1e9, torch.finfo(negative_logits.dtype).min)
        negative_logits.masked_fill_(
            negative_labels.eq(self.negative_labels_ignore_index).unsqueeze(0),
            masked_value,
        )
        negative_logits[
            collision_rows[collisions], collision_columns[collisions]
        ] = masked_value
        return negative_logits

    def _score_group(
        self,
        model_embeddings: torch.Tensor,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
    ) -> torch.Tensor:
        positive_logits = self.logits_callback(
            model_embeddings, positive_labels.unsqueeze(-1)
        )
        scoring_negative_labels = negative_labels.masked_fill(
            negative_labels.eq(self.negative_labels_ignore_index),
            0,
        )
        negative_logits = self.logits_callback(
            model_embeddings, scoring_negative_labels
        )
        return self._loss_from_logits(
            positive_logits,
            negative_logits,
            positive_labels,
            negative_labels,
        )

    def forward(
        self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,  # noqa: ARG002
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        active_groups = self._validate_inputs(
            model_embeddings,
            positive_labels,
            negative_labels,
            target_padding_mask,
        )
        loss = model_embeddings.new_zeros(())
        for group_index in range(active_groups):
            start = group_index * self.logical_batch_size
            end = min(start + self.logical_batch_size, model_embeddings.size(0))
            active_targets = target_padding_mask[start:end].squeeze(-1)
            group_embeddings = model_embeddings[start:end][active_targets]
            group_positive_labels = positive_labels[start:end].squeeze(-1)[
                active_targets
            ]
            if group_embeddings.size(0) == 0:
                msg = "Each active logical group must contain at least one target."
                raise ValueError(msg)
            loss = loss + self._score_group(
                group_embeddings,
                group_positive_labels,
                negative_labels[group_index],
            )
        return loss / self.groups_per_batch
