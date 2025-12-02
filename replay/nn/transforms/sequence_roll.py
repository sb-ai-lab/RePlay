import torch
from typing_extensions import Self

from replay.nn.transforms.base import BaseTransform


class SequenceRollTransform(BaseTransform):
    """
    Rolls the data along axis 1 by the specified amount
    and fills the remaining positions by specified padding value.

    Example:

    .. code-block:: python

        >>> input_tensor = {"item_id": torch.LongTensor([[2, 3, 1]])}
        >>> transform = SequenceRollTransform("item_id", roll=-1, padding_value=10)
        >>> output_tensor = transform(input_tensor)
        >>> print(output_tensor)
        {'item_id': tensor([[ 3,  1, 10]])}

    """

    def __init__(
        self: Self,
        field_name: str,
        roll: int = -1,
        padding_value: int = 0,
    ) -> None:
        """
        :param field_name: Name of the target column from the batch to be rolled.
        :param roll: Number of positions to roll by. Default: ``-1``.
        :param padding_value: The value to use as padding for the sequence. Default: ``0``.
        """
        super().__init__()
        self.field_name = field_name
        self.roll = roll
        self.padding_value = padding_value

    def forward(self: Self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rolled_seq = batch[self.field_name].roll(self.roll, dims=1)

        if self.roll > 0:
            rolled_seq[:, : self.roll, ...] = self.padding_value
        else:
            rolled_seq[:, self.roll :, ...] = self.padding_value

        batch[self.field_name] = rolled_seq
        return batch
