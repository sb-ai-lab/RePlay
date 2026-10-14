import torch

from replay.data.nn import TensorMap

from .login_ce import LogInCESampled


class GroupedLogInCESampled(LogInCESampled):
    """Compute sampled LogInCE with a negative pool per logical batch group.

    The physical batch is divided into consecutive groups of at most
    ``logical_batch_size`` rows. Group ``i`` uses ``negative_labels[i]``; the
    final group may contain fewer rows. This is useful when several smaller
    batches are packed together but each must keep an independent negative
    sampling scope.

    This loss retains the multi-positive objective of
    :class:`~replay.nn.loss.LogInCESampled`.
    ``positive_labels`` and ``target_padding_mask`` therefore have shape
    ``[batch, sequence, num_positives]``. Masked positives do not contribute,
    positive items that occur among the negatives are excluded, and every
    logical group must contain at least one valid target. The result is the mean
    of the group losses, so every active group has equal weight.

    ``negative_labels`` must have shape ``[ceil(B / G), K]``, where ``B`` is the
    physical batch size, ``G`` is ``logical_batch_size``, and ``K`` is the
    number of negatives. Item IDs in each pool must be unique. Use
    :class:`~replay.nn.transform.GroupedUniformNegativeSamplingTransform`
    to satisfy this contract; its ``group_size`` must equal ``logical_batch_size``.
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
        """Calculate the mean LogInCE loss over active logical groups.

        :param model_embeddings: Model output of shape
            ``[batch, sequence, embedding]``.
        :param feature_tensors: Feature mapping accepted for the common RePlay
            loss interface.
        :param positive_labels: Positive item IDs of shape
            ``[batch, sequence, num_positives]``.
        :param negative_labels: One shared negative pool per group, with shape
            ``[ceil(batch / logical_batch_size), num_negatives]``.
        :param padding_mask: Input mask of shape ``[batch, sequence]``.
        :param target_padding_mask: Boolean mask with the same shape as
            ``positive_labels``.
        :return: Scalar mean of the per-group LogInCE losses.
        """
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
