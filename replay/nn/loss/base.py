from collections.abc import Callable
from typing import Protocol, TypedDict

import torch

from replay.data.nn import TensorMap


class LossProto(Protocol):
    """Class-protocol for working with losses inside models"""

    @property
    def logits_callback(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]: ...

    @logits_callback.setter
    def logits_callback(self, func: Callable | None) -> None: ...

    def forward(
        self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor: ...


class SampledLossOutput(TypedDict):
    """A class containing result of the `get_sampled_logits` function in sampled losses"""

    positive_logits: torch.Tensor
    negative_logits: torch.Tensor
    positive_labels: torch.LongTensor
    negative_labels: torch.LongTensor


class SampledLossBase(torch.nn.Module):
    """The base class for calculating sampled losses"""

    negative_labels_ignore_index: int

    @property
    def logits_callback(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]:
        raise NotImplementedError()  # pragma: no cover

    def get_sampled_logits(
        self,
        model_embeddings: torch.Tensor,
        positive_labels: torch.LongTensor,  # [batch_size, seq_len, num_positives]
        negative_labels: torch.LongTensor,  # [num_negatives] or [batch_size, seq_len, num_negatives]
        target_padding_mask: torch.BoolTensor,  # [batch_size, seq_len, num_positives]
    ) -> SampledLossOutput:
        """
        The function of calculating positive and negative logits.
        Based on the model last hidden state, positive and negative labels.

        The function supports the calculation of logits for the case of multi-positive labels
        (there are several labels for each position in the sequence).

        :param model_embeddings: Embeddings from the model. This is usually the last hidden state.
            Expected shape: ``(batch_size, sequence_length, embedding_dim)``
        :param positive_labels: a tensor containing labels with positive events.
            Expected shape: ``(batch_size, sequence_length, num_positives)``
        :param negative_labels: a tensor containing labels with negative events.
            Expected shape:
                - ``(batch_size, sequence_length, num_negatives)``
                - ``(batch_size, num_negatives)``
                - ``(num_negatives)`` - a case where the same negative events are used for the entire batch.
        :param target_padding_mask: Padding mask for ``positive_labels`` (targets).
            ``False`` value indicates that the corresponding ``key`` value will be ignored.
            Expected shape: ``(batch_size, sequence_length, num_positives)``

        :returns: SampledLossOutput. A dictionary containing positive and negative logits with labels.
        """

        initial_positive_labels = positive_labels
        ################## SHAPE CHECKING STAGE START ##################
        batch_size, seq_len, num_positives = positive_labels.size()
        assert target_padding_mask.size() == (batch_size, seq_len, num_positives)
        num_negatives = negative_labels.size(-1)

        if negative_labels.size() == (batch_size, num_negatives):
            # [batch_size, num_negatives] -> [batch_size, 1, num_negatives]
            negative_labels = negative_labels.unsqueeze(1).repeat(1, seq_len, 1)

        if negative_labels.dim() == 3:
            # [batch_size, seq_len, num_negatives] -> [batch_size, seq_len, 1, num_negatives]
            negative_labels = negative_labels.unsqueeze(-2)
            if num_positives != 1:
                # [batch_size, seq_len, num_negatives] -> [batch_size, seq_len, num_positives, num_negatives]
                negative_labels = negative_labels.repeat((1, 1, num_positives, 1))
        assert (
            negative_labels.size() == (batch_size, seq_len, num_positives, num_negatives) or negative_labels.dim() == 1
        )
        ################## SHAPE CHECKING STAGE END ##################

        # Get output embedding for every user event
        embedding_dim = model_embeddings.size(-1)
        assert model_embeddings.size() == (batch_size, seq_len, embedding_dim)

        # [batch_size, seq_len, emb_dim] ->  [batch_size, seq_len, 1, emb_dim]
        model_embeddings = model_embeddings.unsqueeze(-2)
        if num_positives != 1:  # multti positive branch
            model_embeddings = model_embeddings.repeat((1, 1, num_positives, 1))
        assert model_embeddings.size() == (
            batch_size,
            seq_len,
            num_positives,
            embedding_dim,
        )

        # Apply target mask
        # [batch_size, seq_len, num_positives] -> [batch_size, seq_len]
        masked_batch_size = target_padding_mask.sum().item()

        # [batch_size, seq_len, num_positives] -> [masked_batch_size, 1]
        positive_labels = positive_labels[target_padding_mask].unsqueeze(-1)
        assert positive_labels.size() == (masked_batch_size, 1)

        if negative_labels.dim() != 1:
            # [batch_size, seq_len, num_positives, num_negatives] -> [masked_batch_size, num_negatives]
            negative_labels = negative_labels[target_padding_mask]
            assert negative_labels.size() == (masked_batch_size, num_negatives)

        # [batch_size, seq_len, num_positives, emb_dim] -> [masked_batch_size, emb_dim]
        model_embeddings = model_embeddings[target_padding_mask]
        assert model_embeddings.size() == (masked_batch_size, embedding_dim)

        # Get positive and negative logits
        positive_logits = self.logits_callback(model_embeddings, positive_labels)
        assert positive_logits.size() == (masked_batch_size, 1)

        negative_labels_for_lookup = negative_labels.masked_fill(
            negative_labels == self.negative_labels_ignore_index,
            0,
        )
        negative_logits = self.logits_callback(model_embeddings, negative_labels_for_lookup)
        assert negative_logits.size() == (masked_batch_size, num_negatives)

        if num_positives != 1:
            # [batch_size, seq_len, num_positives] -> [batch_size * seq_len]
            masked_target_padding_mask = target_padding_mask.sum(-1).view(-1)
            # [batch_size, seq_len, num_positives] -> [masked_batch_size, num_positives]
            positive_labels = torch.repeat_interleave(
                initial_positive_labels.view(-1, num_positives),
                masked_target_padding_mask,
                dim=0,
            )

        return {
            "positive_logits": positive_logits,
            "negative_logits": negative_logits,
            "positive_labels": positive_labels,
            "negative_labels": negative_labels,
        }


def mask_negative_logits(
    negative_logits: torch.Tensor,
    negative_labels: torch.LongTensor,
    positive_labels: torch.LongTensor,
    negative_labels_ignore_index: int,
    negative_column_lookup: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Assign very small values in negative logits
    for positions where positive labels equal to negative ones.

    :param negative_logits: Logits from the model for ``negative labels``.
        Expected shape: (masked_batch_size, num_negatives)
    :param negative_labels: a tensor containing labels with negative events.
        Expected shape:
            - (masked_batch_size, num_negatives)
            - (num_negatives) - a case where the same negative events are used for the entire batch
    :param positive_labels: a tensor containing labels with positive events.
        Expected shape: (masked_batch_size, num_positives)
    :param negative_labels_ignore_index: padding value for negative labels.
        This may be the case when negative labels
        are formed at the preprocessing level, rather than the negative sampler.
        The index is ignored and does not contribute to the loss.
    :param negative_column_lookup: Optional reusable item-ID-to-logit-column lookup.
        Enables efficient masking for one shared negative pool and one positive
        label per row.

    :returns: Negative logits with modified elements in those positions
        where positive labels are equal to negative ones.
    """

    if negative_column_lookup is not None:
        return _mask_shared_negative_logits(
            negative_logits,
            negative_labels,
            positive_labels,
            negative_labels_ignore_index,
            negative_column_lookup,
        )

    ignored_negatives = negative_labels == negative_labels_ignore_index

    if negative_labels.dim() > 1:
        # [masked_batch_size, num_negatives] -> [masked_batch_size, 1, num_negatives]
        negative_labels = negative_labels.unsqueeze(-2)

    # [masked_batch_size, num_positives] -> [masked_batch_size, num_positives, 1]
    positive_labels = positive_labels.unsqueeze(-1)
    negative_mask = positive_labels == negative_labels  # [masked_batch_size, num_positives, num_negatives]

    # [masked_batch_size, num_positives, num_negatives] -> [masked_batch_size, num_negatives]
    negative_mask = negative_mask.sum(-2).bool()
    negative_logits.masked_fill_(negative_mask | ignored_negatives, -torch.inf)
    return negative_logits


def _mask_shared_negative_logits(
    negative_logits: torch.Tensor,
    negative_labels: torch.LongTensor,
    positive_labels: torch.LongTensor,
    negative_labels_ignore_index: int,
    negative_column_lookup: torch.Tensor,
) -> torch.Tensor:
    """Mask positive collisions in logits for one shared negative pool.

    ``negative_labels`` must be one-dimensional and contain unique non-ignored
    item IDs. ``positive_labels`` must have shape ``(num_targets, 1)``.
    ``negative_column_lookup`` is a reusable one-dimensional integer scratch
    tensor whose length defines the valid item ID range. The function modifies
    both ``negative_logits`` and ``negative_column_lookup`` in place.

    :param negative_logits: Logits with shape ``(num_targets, num_negatives)``.
    :param negative_labels: Unique shared negative item IDs with shape ``(num_negatives,)``.
    :param positive_labels: Positive item IDs with shape ``(num_targets, 1)``.
    :param negative_labels_ignore_index: Value ignored in negative labels.
    :param negative_column_lookup: Reusable item-ID-to-logit-column lookup tensor.
    :returns: ``negative_logits`` with ignored negatives and positive collisions masked.
    """
    if negative_labels.dim() != 1 or positive_labels.dim() != 2 or positive_labels.size(1) != 1:
        msg = "The negative column lookup requires one shared pool and one positive label per row."
        raise ValueError(msg)

    positive_labels = positive_labels.squeeze(-1)
    negative_column_lookup.fill_(-1)
    cardinality = negative_column_lookup.numel()
    valid_negatives = negative_labels.ge(0) & negative_labels.lt(cardinality)
    negative_columns = torch.arange(
        negative_labels.numel(),
        dtype=negative_column_lookup.dtype,
        device=negative_labels.device,
    )
    negative_column_lookup[negative_labels[valid_negatives]] = negative_columns[valid_negatives]

    valid_positives = positive_labels.ge(0) & positive_labels.lt(cardinality)
    safe_positives = positive_labels.clamp(min=0, max=cardinality - 1)
    collision_columns = negative_column_lookup[safe_positives].long()
    collision_rows = torch.arange(positive_labels.numel(), device=positive_labels.device)
    collisions = valid_positives & collision_columns.ge(0)
    negative_logits.masked_fill_(
        negative_labels.eq(negative_labels_ignore_index).unsqueeze(0),
        -torch.inf,
    )
    negative_logits[collision_rows[collisions], collision_columns[collisions]] = -torch.inf
    return negative_logits


def weighted_mean(loss: torch.Tensor, sample_weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculate a weighted mean with a safe denominator."""
    return (loss * sample_weight).sum() / sample_weight.sum().clamp_min(eps)
