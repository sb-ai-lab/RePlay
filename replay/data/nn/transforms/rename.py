import torch
from typing_extensions import Self

from replay.data.nn.transforms.base import BaseTransform


class RenameTransform(BaseTransform):
    """
    Renames specific feature columns into new ones. Changes names in original dict, not creates a new dict.
    """

    def __init__(self: Self, mapping: dict[str, str]) -> None:
        """
        :param mapping: A dict mapping existing names into new ones.
        """
        super().__init__()
        self.mapping = mapping

    def forward(self: Self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for original_name, target_name in self.mapping.items():
            batch[target_name] = batch.pop(original_name)

        return batch
