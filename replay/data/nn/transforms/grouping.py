import torch
from typing_extensions import Self

from replay.data.nn.transforms.base import BaseTransform


class GroupTransform(BaseTransform):
    """
    Composes existing columns into named groups.
    """

    def __init__(self: Self, mapping: dict[str, list[str]]) -> None:
        """
        :param mapping: A dict mapping new names to a list of existing names for grouping.
        """
        super().__init__()
        self.mapping = mapping

    def forward(self: Self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for group_name in self.mapping:
            batch[group_name] = {feature_name: batch[feature_name] for feature_name in self.mapping[group_name]}

        return batch
