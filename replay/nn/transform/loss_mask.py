import torch


class SequenceLossMaskTransform(torch.nn.Module):
    """
    Transform applyes a feature-based loss mask to the target padding mask.

    The loss is usually computed for all non-padded target positions.
    This Transform restricts the loss to only those positions where a given feature matches a specified value
    (for example, only events of a certain type) using the feature-based loss mask.
    This mask is created by selecting a specified value of some feature (float, int or bool type)
    from a batch inside the transform.

    The resulting tensor is supposed to be used as target padding mask (shape
    ``(batch_size, sequence_length)`` or ``(batch_size, sequence_length, num_posititves)`` in multipositive case).
    It is the logical AND between the target padding mask and the feature-based mask.

    Example:

    .. code-block:: python

        >>> input_batch = {
        ...     "target_padding_mask": torch.BoolTensor([[0, 1, 1, 1, 1]]),
        ...     "events_type": torch.LongTensor([0, 3, 2, 1, 2])
        ... }
        >>> transform = SequenceLossMaskTransform(loss_mask_name="events_type", loss_mask_value=2)
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'target_padding_mask': tensor([[False, False,  True, False,  True]]),
         'events_type': tensor([0, 3, 2, 1, 2])}

    """

    def __init__(
        self,
        loss_mask_name: str,
        loss_mask_value: float | bool,
        target_padding_mask_name: str = "target_padding_mask",
    ) -> None:
        """
        :param loss_mask_name: Key name in batch of tensor
            of shape ``(batch_size, sequence_length)`` or ``(batch_size, sequence_length, num_posititves)``
            containing a feature aligned with target positions for loss mask creating.
        :param loss_mask_value: Value used to select which target positions should contribute to the loss for loss mask
            creating.
        :param target_padding_mask_name: Key name in batch of boolean tensor
            of shape ``(batch_size, sequence_length)`` or ``(batch_size, sequence_length, num_posititves)``
            indicating valid (non-padded) target positions. Default: `"target_padding_mask"`.
        """
        super().__init__()
        self.loss_mask_name = loss_mask_name
        self.loss_mask_value = loss_mask_value
        self.target_padding_mask_name = target_padding_mask_name

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        target_mask = output_batch[self.loss_mask_name] == self.loss_mask_value
        output_batch[self.target_padding_mask_name] &= target_mask

        assert output_batch[self.target_padding_mask_name].size() == batch[self.target_padding_mask_name].size()

        return output_batch
