import torch


class SelectTransform(torch.nn.Module):
    """
    Selects the specified features from the batch removing the remaining features.
    Returns batch with selected features.

    Example:

    .. code-block:: python

        >>> input_batch = {"feature_1": torch.tensor([]), "feature_2": torch.tensor([])}
        >>> transform = SelectTransform(["feature_2"])
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'feature_2': tensor([])}

    """

    def __init__(self, feature_names: list[str]) -> None:
        """
        :param feature_names: a list with names of features to be selected.
        """
        super().__init__()
        self._feature_names = feature_names

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v for k, v in batch.items() if k in self._feature_names}
