import torch
from typing_extensions import Self

from replay.nn.transforms.base import BaseTransform


class GroupTransform(BaseTransform):
    """
    Composes existing columns into named groups.

    Example:

    .. code-block:: python

        >>> input_batch = {"item_id": torch.LongTensor([[30, 22, 1]]),
        >>>                 "item_feature": torch.Tensor([[0.3, 0.2, 0.1]])}
        >>> transform = GroupTransform({"tensor_features" : ["item_id", "item_feature"]})
        >>> output_batch = transform(input_batch)
        >>> print(output_batch)
        {'item_id': tensor([[30, 22, 1]]),
        'item_feature': tensor([[0.3, 0.2, 0.1]]),
        'tensor_features': {
            'item_id': tensor([[30, 22, 1]]),
            'item_feature': tensor([[0.3, 0.2, 0.1]])
            }
        }

    """

    def __init__(self: Self, mapping: dict[str, list[str]]) -> None:
        """
        :param mapping: A dict mapping new names to a list of existing names for grouping.
        """
        super().__init__()
        self.mapping = mapping

    def forward(self: Self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for group_name in self.mapping:
            batch[group_name] = {
                feature_name: batch[feature_name]
                for feature_name in self.mapping[group_name]
            }

        return batch
