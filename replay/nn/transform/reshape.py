import torch


class UnsqueezeTransform(torch.nn.Module):
    """
    Unsqueeze a tensor got by specified key from batch along specified dimension.

    Example:

    .. code-block:: python

        >>> input_batch = {"padding_id": torch.BoolTensor([False, True, True])}
        >>> transform = UnsqueezeTransform("padding_id", dim=0)
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'padding_id': tensor([[False,  True,  True]])}

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

        output_batch = {k: v for k, v in batch.items() if k != self.column_name}
        output_batch[self.column_name] = batch[self.column_name].unsqueeze(self.dim)

        return output_batch
