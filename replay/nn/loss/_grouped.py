from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from .base import LossProto


@dataclass(frozen=True)
class PackedGroupedBatch:
    """Dense representation of active targets in logical batch groups."""

    model_embeddings: torch.Tensor
    positive_labels: torch.LongTensor
    target_padding_mask: torch.BoolTensor
    position_mask: torch.BoolTensor


@runtime_checkable
class GroupedLossProto(Protocol):
    """Loss exposing the logical batch boundary used by grouped sampling."""

    logical_batch_size: int


def validate_grouped_loss(loss: LossProto, negative_group_size: int | None) -> None:
    """Validate grouped sampler metadata against the configured loss."""
    if negative_group_size is None:
        return
    if not isinstance(loss, GroupedLossProto):
        msg = "Grouped negative sampling requires a loss with logical_batch_size."
        raise ValueError(msg)
    if negative_group_size != loss.logical_batch_size:
        msg = (
            "The grouped negative sampler and loss must use the same group size. "
            f"Sampler uses {negative_group_size}, loss expects {loss.logical_batch_size}."
        )
        raise ValueError(msg)


def validate_grouped_inputs(
    *,
    loss_name: str,
    model_embeddings: torch.Tensor,
    positive_labels: torch.LongTensor,
    negative_labels: torch.LongTensor,
    target_padding_mask: torch.BoolTensor,
    logical_batch_size: int,
) -> int:
    """Validate shared grouped-loss inputs and return the active group count."""
    if model_embeddings.dim() != 3:
        msg = "model_embeddings must have shape [batch, sequence, embedding]."
        raise ValueError(msg)
    if positive_labels.dim() != 3:
        msg = "positive_labels must have shape [batch, sequence, positives]."
        raise ValueError(msg)
    if positive_labels.shape[:2] != model_embeddings.shape[:2]:
        msg = "positive_labels and model_embeddings must have equal batch and sequence dimensions."
        raise ValueError(msg)
    if target_padding_mask.shape != positive_labels.shape:
        msg = "target_padding_mask must have the same shape as positive_labels."
        raise ValueError(msg)
    if negative_labels.dim() != 2:
        msg = "negative_labels must have shape [num_groups, num_negatives]."
        raise ValueError(msg)
    if model_embeddings.size(0) == 0:
        msg = f"{loss_name} does not support empty batches."
        raise ValueError(msg)

    active_groups = (model_embeddings.size(0) + logical_batch_size - 1) // logical_batch_size
    if negative_labels.size(0) != active_groups:
        msg = f"negative_labels must contain {active_groups} pools, got {negative_labels.size(0)}."
        raise ValueError(msg)
    return active_groups


def pack_grouped_batch(
    *,
    model_embeddings: torch.Tensor,
    positive_labels: torch.LongTensor,
    target_padding_mask: torch.BoolTensor,
    logical_batch_size: int,
    active_groups: int,
) -> PackedGroupedBatch:
    """Pad a partial final group and pack each group's active target positions."""
    batch_size, sequence_length, embedding_dim = model_embeddings.shape
    num_positives = positive_labels.size(-1)
    padded_batch_size = active_groups * logical_batch_size
    padded_rows = padded_batch_size - batch_size
    if padded_rows:
        model_embeddings = _pad_rows(model_embeddings, padded_rows)
        positive_labels = _pad_rows(positive_labels, padded_rows)
        target_padding_mask = _pad_rows(target_padding_mask, padded_rows)

    group_capacity = logical_batch_size * sequence_length
    grouped_embeddings = model_embeddings.reshape(active_groups, group_capacity, embedding_dim)
    grouped_positive_labels = positive_labels.reshape(active_groups, group_capacity, num_positives)
    grouped_target_mask = target_padding_mask.reshape(active_groups, group_capacity, num_positives)
    grouped_position_mask = grouped_target_mask.any(-1)
    target_counts = grouped_position_mask.sum(-1)
    if torch.any(target_counts == 0):
        msg = "Each active logical group must contain at least one target."
        raise ValueError(msg)

    max_targets = int(target_counts.max().item())
    positions = torch.arange(group_capacity, device=grouped_position_mask.device)
    sort_keys = (~grouped_position_mask).long() * group_capacity + positions
    active_first = torch.argsort(sort_keys, dim=-1)[:, :max_targets]
    packed_embeddings = grouped_embeddings.gather(
        1,
        active_first.unsqueeze(-1).expand(-1, -1, embedding_dim),
    )
    packed_positive_labels = grouped_positive_labels.gather(
        1,
        active_first.unsqueeze(-1).expand(-1, -1, num_positives),
    )
    packed_target_mask = grouped_target_mask.gather(
        1,
        active_first.unsqueeze(-1).expand(-1, -1, num_positives),
    )
    packed_position_mask = torch.arange(max_targets, device=target_counts.device).unsqueeze(
        0
    ) < target_counts.unsqueeze(1)
    packed_positive_labels = packed_positive_labels.masked_fill(
        ~packed_position_mask.unsqueeze(-1),
        0,
    )
    return PackedGroupedBatch(
        model_embeddings=packed_embeddings,
        positive_labels=packed_positive_labels,
        target_padding_mask=packed_target_mask,
        position_mask=packed_position_mask,
    )


def evaluate_group_losses(
    group_loss: Callable[..., torch.Tensor],
    *grouped_inputs: torch.Tensor,
) -> torch.Tensor:
    """Evaluate groups with ``vmap`` when available, otherwise sequentially."""
    active_groups = grouped_inputs[0].size(0)
    if active_groups == 1:
        return group_loss(*(input_[0] for input_ in grouped_inputs)).unsqueeze(0)
    vmap = getattr(torch, "vmap", None)
    if vmap is None:
        return torch.stack(
            [group_loss(*(input_[group] for input_ in grouped_inputs)) for group in range(active_groups)]
        )
    return vmap(group_loss, randomness="different")(*grouped_inputs)


def score_sampled_candidates(
    logits_callback: Callable[[torch.Tensor, torch.LongTensor], torch.Tensor],
    model_embeddings: torch.Tensor,
    positive_labels: torch.LongTensor,
    negative_labels: torch.LongTensor,
    negative_labels_ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score one logical group's positive and negative candidates."""
    positive_logits = logits_callback(model_embeddings, positive_labels)
    scoring_negative_labels = negative_labels.masked_fill(
        negative_labels.eq(negative_labels_ignore_index),
        0,
    )
    negative_logits = logits_callback(model_embeddings, scoring_negative_labels)
    return positive_logits, negative_logits


def has_duplicate_negative_labels(
    sorted_negative_labels: torch.LongTensor,
    negative_labels_ignore_index: int,
) -> bool:
    """Return whether a negative pool has repeated non-ignored item IDs."""
    if sorted_negative_labels.size(-1) < 2:
        return False
    repeated_labels = sorted_negative_labels[:, 1:].eq(sorted_negative_labels[:, :-1])
    repeated_labels &= sorted_negative_labels[:, 1:].ne(negative_labels_ignore_index)
    return bool(repeated_labels.any().item())


def find_negative_collision_columns(
    positive_labels: torch.LongTensor,
    sorted_negative_labels: torch.LongTensor,
    sorted_negative_columns: torch.LongTensor,
    positive_labels_mask: torch.BoolTensor,
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """Map every positive ID to its column in its group's negative pool."""
    num_negatives = sorted_negative_labels.size(-1)
    if num_negatives == 0:
        return torch.zeros_like(positive_labels), torch.zeros_like(positive_labels_mask)
    flat_positive_labels = positive_labels.flatten(1)
    insertion_columns = torch.searchsorted(sorted_negative_labels, flat_positive_labels)
    has_candidate = insertion_columns.lt(num_negatives)
    safe_insertion_columns = insertion_columns.clamp(max=num_negatives - 1)
    matched_negatives = sorted_negative_labels.gather(1, safe_insertion_columns)
    collision_columns = sorted_negative_columns.gather(1, safe_insertion_columns)
    collisions = positive_labels_mask.flatten(1) & has_candidate & matched_negatives.eq(flat_positive_labels)
    return collision_columns.reshape_as(positive_labels), collisions.reshape_as(positive_labels)


def mask_group_negative_logits(
    negative_logits: torch.Tensor,
    positive_labels: torch.LongTensor,
    positive_labels_mask: torch.BoolTensor,
    negative_labels: torch.LongTensor,
    negative_labels_ignore_index: int,
) -> torch.Tensor:
    """Mask every positive collision for one logical group.

    This exact fallback handles repeated IDs in a negative pool. The regular
    grouped-loss path avoids the target-by-negative comparison matrix because
    the grouped sampler draws unique IDs without replacement.
    """
    if positive_labels.dim() == 1:
        positive_labels = positive_labels.unsqueeze(-1)
        positive_labels_mask = positive_labels_mask.unsqueeze(-1)
    collisions = (positive_labels.unsqueeze(-1).eq(negative_labels) & positive_labels_mask.unsqueeze(-1)).any(dim=-2)
    ignored_negatives = negative_labels.eq(negative_labels_ignore_index).unsqueeze(0)
    return negative_logits.masked_fill(collisions | ignored_negatives, -torch.inf)


def mask_unique_group_negative_logits(
    negative_logits: torch.Tensor,
    negative_labels: torch.LongTensor,
    collision_columns: torch.LongTensor,
    collisions: torch.BoolTensor,
    negative_labels_ignore_index: int,
) -> torch.Tensor:
    """Mask collisions in a unique negative pool without a comparison tensor."""
    negative_logits = negative_logits.masked_fill(
        negative_labels.eq(negative_labels_ignore_index).unsqueeze(0),
        -torch.inf,
    )
    if negative_logits.size(-1) == 0:
        return negative_logits

    if collision_columns.dim() == 1:
        collision_columns = collision_columns.unsqueeze(-1)
        collisions = collisions.unsqueeze(-1)
        selected_logits = negative_logits.gather(1, collision_columns)
        selected_logits = selected_logits.masked_fill(collisions, -torch.inf)
        return negative_logits.scatter(1, collision_columns, selected_logits)

    sentinel_column = negative_logits.size(-1)
    negative_logits = torch.cat(
        (negative_logits, negative_logits.new_zeros((negative_logits.size(0), 1))),
        dim=-1,
    )
    collision_columns = collision_columns.masked_fill(~collisions, sentinel_column)
    return negative_logits.scatter(1, collision_columns, -torch.inf)[:, :-1]


def mask_group_candidates(
    negative_logits: torch.Tensor,
    positive_labels: torch.LongTensor,
    positive_labels_mask: torch.BoolTensor,
    negative_labels: torch.LongTensor,
    negative_labels_ignore_index: int,
    collision_map: tuple[torch.LongTensor, torch.BoolTensor] | None = None,
) -> torch.Tensor:
    """Apply the exact duplicate fallback or the fast unique-pool mask."""
    if collision_map is None:
        return mask_group_negative_logits(
            negative_logits,
            positive_labels,
            positive_labels_mask,
            negative_labels,
            negative_labels_ignore_index,
        )
    collision_columns, collisions = collision_map
    return mask_unique_group_negative_logits(
        negative_logits,
        negative_labels,
        collision_columns,
        collisions,
        negative_labels_ignore_index,
    )


def _pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    return torch.cat((tensor, tensor.new_zeros((rows, *tensor.shape[1:]))), dim=0)
