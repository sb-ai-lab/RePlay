import torch
from typing_extensions import Self

from replay.nn.transforms.base import BaseTransform


class RenameTransform(BaseTransform):
    """
    Renames specific feature columns into new ones. Changes names in original dict, not creates a new dict.
    Example:

    .. code-block:: python

        >>> input_batch = {"item_id_mask": torch.BoolTensor([False, True, True])}
        >>> transform = RenameTransform({"item_id_mask" : "padding_id"})
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'padding_id': tensor([False,  True,  True])}

    """

    def __init__(self: Self, mapping: dict[str, str]) -> None:
        """
        :param mapping: A dict mapping existing names into new ones.
        """
        super().__init__()
        self.mapping = mapping

    def forward(self: Self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = {}

        for original_name, tensor in batch.items():
            target_name = self.mapping.get(original_name, original_name)
            output_batch[target_name] = tensor

        return output_batch
