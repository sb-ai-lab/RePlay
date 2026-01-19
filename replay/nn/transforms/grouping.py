import torch
from typing_extensions import Self

from replay.nn.transforms.base import BaseTransform


class GroupTransform(BaseTransform):
    """
    Combines existing tensors from a batch moving them to the common groups.
    The name of the shared keys and the keys to be moved are specified in ``mapping``.

    Example:

    .. code-block:: python

        >>> input_batch = {
        ...     "item_id": torch.LongTensor([[30, 22, 1]]),
        ...     "item_feature": torch.LongTensor([[1, 11, 11]])
        ... }
        >>> transform = GroupTransform({"feature_tensors" : ["item_id", "item_feature"]})
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'feature_tensors': {'item_id': tensor([[30, 22,  1]]),
        'item_feature': tensor([[ 1, 11, 11]])}}

    """

    def __init__(self: Self, mapping: dict[str, list[str]]) -> None:
        """
        :param mapping: A dict mapping new names to a list of existing names for grouping.
        """
        super().__init__()
        self.mapping = mapping
        self._grouped_keys = set().union(*mapping.values())

    def forward(self: Self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = {k: v for k, v in batch.items() if k not in self._grouped_keys}

        for group_name, feature_names in self.mapping.items():
            output_batch[group_name] = {name: batch[name] for name in feature_names if name in batch}

        return output_batch
