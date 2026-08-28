import torch

from replay.data.nn import TensorMap

from ._grouped import (
    evaluate_group_losses,
    find_negative_collision_columns,
    has_duplicate_negative_labels,
    mask_group_candidates,
    pack_grouped_batch,
    score_sampled_candidates,
    validate_grouped_inputs,
)
from .base import mask_negative_logits
from .ce import CESampled


class GroupedCESampled(CESampled):
    """Sampled CE with one independent negative pool per logical batch group.

    The loss is intended for physical batches that pack several logical
    microbatches while retaining an independent negative pool for every
    logical group. All active groups are evaluated together with
    :func:`torch.vmap` when it is available; older supported PyTorch versions
    use a sequential compatibility fallback.

    Use the loss with
    :class:`~replay.nn.transform.GroupedUniformNegativeSamplingTransform`. The
    sampler adds its group size to the training batch, and compatible models
    validate it against the loss so mismatched boundaries are rejected before
    logits are calculated.

    The grouped sampler draws pools without replacement. For manually supplied
    pools with repeated non-ignored IDs, the loss preserves the exact sampled-CE
    semantics with a slower, memory-intensive collision-masking fallback.

    With label smoothing, ignored negatives and sampled negatives that match a
    positive label are excluded from both the softmax and smoothing distribution.

    On the vectorized path, the logits callback must be compatible with
    :func:`torch.vmap`. In particular, training-time modules that update buffers,
    such as :class:`torch.nn.BatchNorm1d`, are unsupported.

    A final partial physical batch may contain only one logical group. That
    boundary case is evaluated directly and is equivalent to :class:`CESampled`.
    """

    def __init__(
        self,
        logical_batch_size: int,
        cardinality: int | None = None,
        negative_labels_ignore_index: int = -100,
        **kwargs,
    ) -> None:
        """
        :param logical_batch_size: Number of rows that share one negative pool.
            It must exactly match the ``group_size`` of the grouped sampler.
        :param cardinality: Optional catalog size. A single logical group uses
            it for the original reusable item-ID lookup; packed groups use a
            sorted collision lookup to avoid allocating a dense
            ``num_groups x cardinality`` scratch tensor.
        :param negative_labels_ignore_index: Value ignored in negative labels.
        :param kwargs: Arguments passed to :class:`torch.nn.CrossEntropyLoss`.
        """
        if logical_batch_size <= 0:
            msg = "The logical_batch_size parameter must be positive."
            raise ValueError(msg)
        if cardinality is not None and cardinality <= 0:
            msg = "The cardinality parameter must be positive."
            raise ValueError(msg)
        super().__init__(negative_labels_ignore_index=negative_labels_ignore_index, **kwargs)
        if self._loss.reduction != "mean":
            msg = "GroupedCESampled supports only mean reduction."
            raise ValueError(msg)
        self.logical_batch_size = logical_batch_size
        self.cardinality = cardinality
        self.register_buffer("_negative_column_lookup", None, persistent=False)

    def _single_group_dense_loss(
        self,
        model_embeddings: torch.Tensor,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
    ) -> torch.Tensor:
        assert self.cardinality is not None  # Narrowed by _forward_single_group.
        lookup = self._negative_column_lookup
        if lookup is None or lookup.device != negative_labels.device or lookup.numel() != self.cardinality:
            lookup = torch.empty(
                self.cardinality,
                dtype=torch.int32,
                device=negative_labels.device,
            )
            self._negative_column_lookup = lookup

        positive_logits, negative_logits = score_sampled_candidates(
            self.logits_callback,
            model_embeddings,
            positive_labels.unsqueeze(-1),
            negative_labels,
            self.negative_labels_ignore_index,
        )
        negative_logits = mask_negative_logits(
            negative_logits,
            negative_labels,
            positive_labels.unsqueeze(-1),
            self.negative_labels_ignore_index,
            negative_column_lookup=lookup,
        )
        logits = torch.cat((positive_logits, negative_logits), dim=-1)
        target = torch.zeros_like(positive_labels)
        return self._compute_loss(logits, target)

    def _forward_single_group(
        self,
        model_embeddings: torch.Tensor,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        active_targets = target_padding_mask.squeeze(-1)
        group_embeddings = model_embeddings[active_targets]
        group_positive_labels = positive_labels.squeeze(-1)[active_targets]
        if group_embeddings.size(0) == 0:
            msg = "Each active logical group must contain at least one target."
            raise ValueError(msg)

        position_mask = torch.ones_like(group_positive_labels, dtype=torch.bool)
        collision_columns = None
        collisions = None
        if negative_labels.size(-1) > 0:
            sorted_negative_labels, sorted_negative_columns = torch.sort(negative_labels, dim=-1)
            if has_duplicate_negative_labels(sorted_negative_labels, self.negative_labels_ignore_index):
                return self._group_loss(
                    group_embeddings,
                    group_positive_labels,
                    negative_labels.squeeze(0),
                    position_mask,
                )
            if self.cardinality is not None:
                return self._single_group_dense_loss(
                    group_embeddings,
                    group_positive_labels,
                    negative_labels.squeeze(0),
                )
            collision_columns, collisions = find_negative_collision_columns(
                group_positive_labels.unsqueeze(0),
                sorted_negative_labels,
                sorted_negative_columns,
                position_mask.unsqueeze(0),
            )
            collision_columns = collision_columns.squeeze(0)
            collisions = collisions.squeeze(0)

        return self._group_loss(
            group_embeddings,
            group_positive_labels,
            negative_labels.squeeze(0),
            position_mask,
            collision_columns,
            collisions,
        )

    def _group_loss(
        self,
        model_embeddings: torch.Tensor,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        position_mask: torch.BoolTensor,
        collision_columns: torch.LongTensor | None = None,
        collisions: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        positive_logits, negative_logits = score_sampled_candidates(
            self.logits_callback,
            model_embeddings,
            positive_labels.unsqueeze(-1),
            negative_labels,
            self.negative_labels_ignore_index,
        )
        negative_logits = mask_group_candidates(
            negative_logits,
            positive_labels,
            position_mask,
            negative_labels,
            self.negative_labels_ignore_index,
            None if collision_columns is None else (collision_columns, collisions),
        )
        logits = torch.cat((positive_logits, negative_logits), dim=-1)
        target = torch.zeros_like(positive_labels).masked_fill(~position_mask, self._loss.ignore_index)
        return self._compute_loss(logits, target)

    def forward(
        self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,  # noqa: ARG002
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        if positive_labels.dim() != 3 or positive_labels.size(-1) != 1:
            msg = "GroupedCESampled requires exactly one positive label per sequence position."
            raise ValueError(msg)
        active_groups = validate_grouped_inputs(
            loss_name=self.__class__.__name__,
            model_embeddings=model_embeddings,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            target_padding_mask=target_padding_mask,
            logical_batch_size=self.logical_batch_size,
        )
        if active_groups == 1:
            return self._forward_single_group(
                model_embeddings,
                positive_labels,
                negative_labels,
                target_padding_mask,
            )

        packed = pack_grouped_batch(
            model_embeddings=model_embeddings,
            positive_labels=positive_labels,
            target_padding_mask=target_padding_mask,
            logical_batch_size=self.logical_batch_size,
            active_groups=active_groups,
        )
        packed_positive_labels = packed.positive_labels.squeeze(-1)
        sorted_negative_labels, sorted_negative_columns = torch.sort(negative_labels, dim=-1)
        if has_duplicate_negative_labels(sorted_negative_labels, self.negative_labels_ignore_index):
            group_losses = evaluate_group_losses(
                self._group_loss,
                packed.model_embeddings,
                packed_positive_labels,
                negative_labels,
                packed.position_mask,
            )
            return group_losses.mean()
        collision_columns, collisions = find_negative_collision_columns(
            packed_positive_labels,
            sorted_negative_labels,
            sorted_negative_columns,
            packed.position_mask,
        )
        group_losses = evaluate_group_losses(
            self._group_loss,
            packed.model_embeddings,
            packed_positive_labels,
            negative_labels,
            packed.position_mask,
            collision_columns,
            collisions,
        )
        return group_losses.mean()
