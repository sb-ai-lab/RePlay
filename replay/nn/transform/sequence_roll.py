import torch


class SequenceRollTransform(torch.nn.Module):
    """
    Rolls the data along axis 1 by the specified amount
    and fills the remaining positions by specified padding value.

    Example:

    .. code-block:: python

        >>> input_tensor = {"item_id": torch.LongTensor([[2, 3, 1]])}
        >>> transform = SequenceRollTransform("item_id", roll=-1, padding_value=10)
        >>> output_tensor = transform(input_tensor)
        >>> output_tensor
        {'item_id': tensor([[ 3,  1, 10]])}

    """

    def __init__(
        self,
        feature_name: str,
        roll: int = -1,
        padding_value: int = 0,
    ) -> None:
        """
        :param feature_name: Name of the tensor from the batch to be rolled.
        :param roll: Number of positions to roll by. Default: ``-1``.
        :param padding_value: The value to use as padding for the sequence. Default: ``0``.
        """
        super().__init__()
        self.feature_name = feature_name
        self.roll = roll
        self.padding_value = padding_value

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = {k: v for k, v in batch.items() if k != self.feature_name}

        rolled_seq = batch[self.feature_name].roll(self.roll, dims=1)

        if self.roll > 0:
            rolled_seq[:, : self.roll, ...] = self.padding_value
        else:
            rolled_seq[:, self.roll :, ...] = self.padding_value

        output_batch[self.feature_name] = rolled_seq
        return output_batch
