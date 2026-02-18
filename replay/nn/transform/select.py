import torch


class SelectTransform(torch.nn.Module):
    """
    Selects the specified features from the batch removing the remaining features.
    Returns batch with selected features.

    Features to be selected may be specified in list in 3 options:
      - str: "k"                      (depth 1)
      - tuple of 1: ("k",)            (depth 1)
      - tuple of 2: ("k1", "k2")      (depth 2)

    Example:

    .. code-block:: python

        >>> input_batch = {
        ...     "feature_1": torch.tensor([1]),
        ...     "feature_2": torch.tensor([2]),
        ...     "feature_group": {
        ...         "feature_3": torch.tensor([3]),
        ...         "feature_4": torch.tensor([4]),
        ...     },
        ... }
        >>> transform = SelectTransform(["feature_1", ("feature_2",), ("feature_group", "feature_3")])
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'feature_1': tensor([1]),
         'feature_2': tensor([2]),
         'feature_group': {'feature_3': tensor([3])}}

    """

    def __init__(self, feature_names: list[str | tuple[str]]) -> None:
        """
        :param feature_names: a list with names of features to be selected.
        """
        super().__init__()
        self._feature_names = feature_names

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = {}

        for key in self._feature_names:
            key = (key,) if isinstance(key, str) else key

            if len(key) == 1:
                key = key[0]
                if key not in batch:
                    msg = f"Key={key} doesn't exist in input batch."
                    raise KeyError()

                output_batch[key] = batch[key]

            elif len(key) == 2:
                key, subkey = key

                if key not in batch:
                    msg = f"Key={key} doesn't exist in input batch."
                    raise KeyError(msg)
                elif subkey not in batch[key]:
                    msg = f"Subkey={subkey} doesn't exist in subbatch by key={key}."
                    raise KeyError(msg)

                out_tensor = batch[key][subkey]
                if key in output_batch:
                    output_batch[key][subkey] = out_tensor
                else:
                    output_batch[key] = {subkey: out_tensor}
            else:
                msg = f"Keys of depth 1 or 2 is only supported, got key={key}"
                raise NotImplementedError(msg)

        return output_batch
