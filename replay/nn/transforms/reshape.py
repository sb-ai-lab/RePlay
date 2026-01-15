import torch

from replay.nn.transforms.base import BaseTransform


class UnsqueezeTransform(BaseTransform):
    """
    Unsqueeze specified tensor along specified dimension.

    Example:

    .. code-block:: python

        >>> input_tensor = {"padding_id": torch.BoolTensor([False, True, True])}
        >>> transform = UnsqueezeTransform("padding_id", dim=-1)
        >>> output_tensor = transform(input_tensor)
        >>> output_tensor
        {'padding_id': tensor([[False],
         [ True],
         [ True]])}

    """

    def __init__(self, column_name: str, dim: int) -> None:
        """
        :param column_name: Name of tensor to be unsqueezed.
        :param dim: Dimension along which tensor will be unsqueezed.
        """
        super().__init__()
        self.column_name = column_name
        self.dim = dim

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.dim > batch[self.column_name].ndim - 1:
            msg = (
                "The dim parameter is incorrect."
                f"Expected unsqueezing by {self.dim} dimension,"
                f"but got the tensor with {batch[self.column_name].ndim} dimensions."
            )
            raise ValueError(msg)

        batch[self.column_name].unsqueeze_(self.dim)

        return batch
