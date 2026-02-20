from typing import Literal

import torch


class EqualityMaskTransform(torch.nn.Module):
    """
    Transform applyes a feature-based mask to the existing boolean mask (key ``mask_name`` in batch).

    The feature-based mask is created by feature (key ``feature_name``) from a batch inside the transform.
    Mask contains True values at the only those positions where a given feature matches a specified value
    ``equality_value`` (for example, only events of a certain type).

    Then, the specified logical operation ``mode`` is applyed
    between the existing boolean mask and the feature-based mask.

    The resulting tensor is supposed to be used as the previous boolean mask
    and setted to the its previous key ``mask_name``.

    Example:

    .. code-block:: python

        >>> input_batch = {
        ...     "target_padding_mask": torch.BoolTensor([[0, 1, 1, 1, 1]]),
        ...     "events_type": torch.LongTensor([0, 3, 2, 1, 2])
        ... }
        >>> transform = EqualityMaskTransform(feature_name="events_type", equality_value=2, mode="and")
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'target_padding_mask': tensor([[False, False,  True, False,  True]]),
        'events_type': tensor([0, 3, 2, 1, 2])}

    """

    def __init__(
        self,
        feature_name: str,
        equality_value: float | bool,
        mode: Literal["and", "or", "xor"] = "and",
        mask_name: str = "target_padding_mask",
    ) -> None:
        """
        :param feature_name: Key name in batch of tensor containing a feature for mask creating.
        :param equality_value: Value used to select which positions should be non-padded.
        :param mode: type of logical operation to be applyed to ``mask_name`` tensor and created mask.
            Default: `"and"`.
        :param mask_name: Key name in batch of boolean tensor of shape indicating valid (non-padded) positions.
            Default: `"target_padding_mask"`.
        """
        super().__init__()

        if mode not in ["and", "or", "xor"]:
            msg = f"Mode={mode} is not supported. Possible values are 'and', 'or', 'xor'."
            raise ValueError(msg)

        if mode == "and":
            self.logical_op = torch.logical_and
        elif mode == "or":
            self.logical_op = torch.logical_or
        elif mode == "xor":
            self.logical_op = torch.logical_xor

        self.feature_name = feature_name
        self.equality_value = equality_value
        self.mask_name = mask_name

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        mask_modification = output_batch[self.feature_name] == self.equality_value

        output_batch[self.mask_name] = self.logical_op(output_batch[self.mask_name], mask_modification)

        assert output_batch[self.mask_name].size() == batch[self.mask_name].size()
        return output_batch
