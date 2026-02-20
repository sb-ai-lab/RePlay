import torch


class CopyTransform(torch.nn.Module):
    """
    Copies a set of columns according to the provided mapping.
    All copied columns are detached from the graph to prevent erroneous
    differentiation.

    Example:

    .. code-block:: python

        >>> input_batch = {"item_id_mask": torch.BoolTensor([False, True, True])}
        >>> transform = CopyTransform({"item_id_mask" : "padding_id"})
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'item_id_mask': tensor([False,  True,  True]),
        'padding_id': tensor([False,  True,  True])}

    """

    def __init__(self, mapping: dict[str, str]) -> None:
        """
        :param mapping: A dictionary maps which source tensors will be copied into the batch with new names.
            Tensors with new names will be copies of original ones, original tensors are stayed in batch.
        """
        super().__init__()
        self.mapping = mapping

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())
        output_batch |= {
            out_column: output_batch[in_column].clone().detach() for in_column, out_column in self.mapping.items()
        }
        return output_batch
