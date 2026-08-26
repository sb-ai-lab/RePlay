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
from .login_ce import LogInCESampled


class GroupedLogInCESampled(LogInCESampled):
    """Sampled LogInCE with one negative pool per logical batch group.

    The loss keeps the multi-positive objective of :class:`LogInCESampled`.
    It is useful when several former microbatches are packed into one physical
    batch while every logical microbatch must retain an independent negative
    sample pool. Use it with
    :class:`~replay.nn.transform.GroupedUniformNegativeSamplingTransform` and
    set ``logical_batch_size`` to the sampler's ``group_size``. The sampler
    adds that value to the training batch, and compatible models validate it
    against the loss so mismatched boundaries are rejected before logits are
    calculated.

    All active groups are evaluated together with :func:`torch.vmap` when it
    is available; older supported PyTorch versions use a sequential
    compatibility fallback. A partial final physical batch may contain one
    logical group, which is evaluated directly and is equivalent to
    :class:`LogInCESampled`.

    The grouped sampler draws pools without replacement. For manually supplied
    pools with repeated non-ignored IDs, the loss preserves the exact sampled
    objective with a slower, memory-intensive collision-masking fallback.

    On the vectorized path, the logits callback must be compatible with
    :func:`torch.vmap`. In particular, training-time modules that update buffers,
    such as :class:`torch.nn.BatchNorm1d`, are unsupported.

    Example:

    .. code-block:: python

        from replay.data.nn import ParquetModule
        from replay.nn.loss import GroupedLogInCESampled
        from replay.nn.sequential import SasRec
        from replay.nn.transform import GroupedUniformNegativeSamplingTransform
        from replay.nn.transform.template import make_default_sasrec_transforms

        logical_batch_size = 128
        packed_batch_size = 5 * logical_batch_size

        negative_sampler = GroupedUniformNegativeSamplingTransform(
            cardinality=num_items,
            num_negative_samples=20_000,
            group_size=logical_batch_size,
        )
        loss = GroupedLogInCESampled(logical_batch_size=logical_batch_size)

        transforms = make_default_sasrec_transforms(tensor_schema)
        transforms["train"].append(negative_sampler)
        datamodule = ParquetModule(
            metadata=metadata,
            transforms=transforms,
            batch_size=packed_batch_size,
            train_path=train_path,
            validate_path=validate_path,
        )
        model = SasRec(body=sasrec_body, loss=loss)
    """

    def __init__(
        self,
        logical_batch_size: int,
        log_epsilon: float = 1e-6,
        clamp_border: float = 100.0,
        negative_labels_ignore_index: int = -100,
    ) -> None:
        """
        :param logical_batch_size: Number of rows that share one negative pool.
            It must exactly match the ``group_size`` of the grouped negative
            sampler.
        :param log_epsilon: Correction to avoid zero in the logarithm.
        :param clamp_border: Absolute bound used to clamp the loss tensor.
        :param negative_labels_ignore_index: Value ignored in negative labels.
        """
        if logical_batch_size <= 0:
            msg = "The logical_batch_size parameter must be positive."
            raise ValueError(msg)
        super().__init__(
            log_epsilon=log_epsilon,
            clamp_border=clamp_border,
            negative_labels_ignore_index=negative_labels_ignore_index,
        )
        self.logical_batch_size = logical_batch_size

    def forward(
        self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,  # noqa: ARG002
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        active_groups = validate_grouped_inputs(
            loss_name=self.__class__.__name__,
            model_embeddings=model_embeddings,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            target_padding_mask=target_padding_mask,
            logical_batch_size=self.logical_batch_size,
        )
        packed = pack_grouped_batch(
            model_embeddings=model_embeddings,
            positive_labels=positive_labels,
            target_padding_mask=target_padding_mask,
            logical_batch_size=self.logical_batch_size,
            active_groups=active_groups,
        )
        if negative_labels.size(-1) == 0:
            # Keep both towers in the autograd graph for distributed training.
            positive_logits = self.logits_callback(
                packed.model_embeddings[0],
                packed.positive_labels[0],
            )
            return (packed.model_embeddings.sum() + positive_logits.sum()) * 0

        sorted_negative_labels, sorted_negative_columns = torch.sort(negative_labels, dim=-1)
        if has_duplicate_negative_labels(sorted_negative_labels, self.negative_labels_ignore_index):
            group_losses = evaluate_group_losses(
                self._group_loss,
                packed.model_embeddings,
                packed.positive_labels,
                negative_labels,
                packed.target_padding_mask,
                packed.position_mask,
            )
            return group_losses.mean()
        collision_columns, collisions = find_negative_collision_columns(
            packed.positive_labels,
            sorted_negative_labels,
            sorted_negative_columns,
            packed.target_padding_mask,
        )
        group_losses = evaluate_group_losses(
            self._group_loss,
            packed.model_embeddings,
            packed.positive_labels,
            negative_labels,
            packed.target_padding_mask,
            packed.position_mask,
            collision_columns,
            collisions,
        )
        return group_losses.mean()

    def _group_loss(
        self,
        model_embeddings: torch.Tensor,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        target_padding_mask: torch.BoolTensor,
        position_mask: torch.BoolTensor,
        collision_columns: torch.LongTensor | None = None,
        collisions: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        positive_logits, negative_logits = score_sampled_candidates(
            self.logits_callback,
            model_embeddings,
            positive_labels,
            negative_labels,
            self.negative_labels_ignore_index,
        )
        negative_logits = mask_group_candidates(
            negative_logits,
            positive_labels,
            target_padding_mask,
            negative_labels,
            self.negative_labels_ignore_index,
            None if collision_columns is None else (collision_columns, collisions),
        )

        max_values = torch.maximum(
            positive_logits.max(-1, keepdim=True).values,
            negative_logits.max(-1, keepdim=True).values,
        )
        positive_values = (torch.exp(positive_logits - max_values) * target_padding_mask).sum(-1)
        negative_values = torch.exp(negative_logits - max_values).sum(-1)
        positive_values = positive_values.masked_fill(~position_mask, 1)
        negative_values = negative_values.masked_fill(~position_mask, 0)
        probabilities = positive_values / (positive_values + negative_values)
        losses = -torch.clamp(
            torch.log(probabilities + self.log_epsilon),
            -self.clamp_border,
            self.clamp_border,
        )
        return losses.masked_fill(~position_mask, 0).sum() / position_mask.sum()
