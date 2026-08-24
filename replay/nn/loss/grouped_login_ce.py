from typing import TYPE_CHECKING

import torch

from replay.data.nn import TensorMap

from .login_ce import LogInCESampled

if TYPE_CHECKING:
    from replay.nn.transform import GroupedUniformNegativeSamplingTransform


class GroupedLogInCESampled(LogInCESampled):
    """Sampled LogInCE with one negative pool per logical batch group.

    The loss keeps the multi-positive objective of :class:`LogInCESampled`.
    It is useful when several former microbatches are packed into one physical
    batch while every logical microbatch must retain an independent negative
    sample pool. Prefer :meth:`from_negative_sampler` when using
    :class:`~replay.nn.transform.GroupedUniformNegativeSamplingTransform`; it
    derives the logical group size from the sampler and prevents configuration
    drift between the two components.

    Use this loss together with packed training batches when every logical batch
    must keep its own sampled-negative pool. The physical batch may end with a
    partial logical group. Do not pair the low-level
    ``GroupedLogInCESampled(logical_batch_size=...)`` constructor with a
    separately configured sampler unless both sizes are guaranteed to be
    identical.

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
        loss = GroupedLogInCESampled.from_negative_sampler(negative_sampler)

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
            sampler. Prefer :meth:`from_negative_sampler` instead of specifying
            the same value twice.
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

    @classmethod
    def from_negative_sampler(
        cls,
        negative_sampler: "GroupedUniformNegativeSamplingTransform",
        *,
        log_epsilon: float = 1e-6,
        clamp_border: float = 100.0,
        negative_labels_ignore_index: int = -100,
    ) -> "GroupedLogInCESampled":
        """Create a loss whose logical group size is derived from its sampler.

        This is the recommended construction path because it makes the sampler
        the single source of truth for logical batch boundaries.

        :param negative_sampler: Grouped sampler used in the training transform pipeline.
        :param log_epsilon: Correction to avoid zero in the logarithm.
        :param clamp_border: Absolute bound used to clamp the loss tensor.
        :param negative_labels_ignore_index: Value ignored in negative labels.
        :return: Configured grouped sampled LogInCE loss.
        """
        from replay.nn.transform import GroupedUniformNegativeSamplingTransform

        if not isinstance(negative_sampler, GroupedUniformNegativeSamplingTransform):
            msg = "negative_sampler must be a GroupedUniformNegativeSamplingTransform instance."
            raise TypeError(msg)
        return cls(
            logical_batch_size=negative_sampler.group_size,
            log_epsilon=log_epsilon,
            clamp_border=clamp_border,
            negative_labels_ignore_index=negative_labels_ignore_index,
        )

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
            msg = "GroupedLogInCESampled does not support empty batches."
            raise ValueError(msg)

        active_groups = (model_embeddings.size(0) + self.logical_batch_size - 1) // self.logical_batch_size
        if negative_labels.size(0) != active_groups:
            msg = f"negative_labels must contain {active_groups} pools, got {negative_labels.size(0)}."
            raise ValueError(msg)
        return active_groups

    def forward(
        self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        active_groups = self._validate_inputs(
            model_embeddings,
            positive_labels,
            negative_labels,
            target_padding_mask,
        )
        group_losses = []
        for group_index in range(active_groups):
            start = group_index * self.logical_batch_size
            end = min(start + self.logical_batch_size, model_embeddings.size(0))
            group_target_mask = target_padding_mask[start:end]
            if not group_target_mask.any():
                msg = "Each active logical group must contain at least one target."
                raise ValueError(msg)
            group_losses.append(
                super().forward(
                    model_embeddings[start:end],
                    feature_tensors,
                    positive_labels[start:end],
                    negative_labels[group_index],
                    padding_mask[start:end],
                    group_target_mask,
                )
            )
        return torch.stack(group_losses).mean()
