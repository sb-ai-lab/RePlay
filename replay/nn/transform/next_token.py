import torch

from replay.data.nn.parquet.impl.masking import DEFAULT_MASK_POSTFIX


class NextTokenTransform(torch.nn.Module):
    """
    For the tensor specified by the key `label_name` (typically “item_id”) in the batch, this transform creates
    a corresponding "labels" tensor with a key ``out_feature_name`` in the batch, shifted forward
    by the specified ``shift`` value. This "labels" tensor is a target that model predicts.
    A padding mask for “labels” is also created. For all features excepted `ignore`,
    last ``shift`` elements are truncated.

    This transform is required for the sequential models in order to optimize the next token prediction task.

    **WARNING**: In order to facilitate the shifting, this transform
    requires extra elements in the sequence. Therefore, when utilizing this
    transform, ensure you're reading at least ``sequence_length`` + ``shift``
    elements from your dataset. The resulting batch will have the relevant fields
    trimmed to ``sequence_length``.

    Example:

    .. code-block:: python

        >>> input_batch = {
        ...     "user_id": torch.LongTensor([111]),
        ...     "item_id": torch.LongTensor([[5, 0, 7, 4]]),
        ...     "item_id_mask": torch.BoolTensor([[0, 1, 1, 1]])
        ... }
        >>> transform = NextTokenTransform(label_name="item_id", shift=1, ignore="user_id")
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'user_id': tensor([111]),
        'item_id': tensor([[5, 0, 7]]),
        'item_id_mask': tensor([[False,  True,  True]]),
        'positive_labels': tensor([[0, 7, 4]]),
        'positive_labels_mask': tensor([[True, True, True]])}

    """

    def __init__(
        self,
        label_name: str,
        shift: int = 1,
        ignore: list[str] | str | None = None,
        out_feature_name: str = "positive_labels",
        mask_postfix: str = DEFAULT_MASK_POSTFIX,
    ) -> None:
        """
        :param label_name: A name of target feature tensor to convert into labels.
        :param shift: The number of sequence items to shift by. Default: `1`.
        :param ignore: Names of keys in batch be excepted from the shifting and will be left unchanged.
        :param out_feature_name: The name of the resulting feature in a batch. Default: `"positive_labels"`.
        :param mask_postfix: a postfix to append to the mask feature corresponding to the resulting feature.
            Default: ``"_mask"``.
        """
        super().__init__()
        self.label_name = label_name
        self.shift = shift
        self.ignore = ignore if isinstance(ignore, list) else [ignore] if isinstance(ignore, str) else []
        self.out_feature_name = out_feature_name
        self.mask_postfix = mask_postfix

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if batch[self.label_name].dim() < 2:
            msg = (
                f"Transform expects batch feature {self.label_name} to be sequential "
                f"but tensor of shape {batch[self.label_name].shape} found."
            )
            raise ValueError(msg)

        max_len = batch[self.label_name].shape[1]
        if self.shift >= max_len:
            msg = (
                f"Transform with shift={self.shift} cannot be applied to sequences of length {max_len}."
                "Decrease value of `shift` parameter in transform"
            )
            raise ValueError(msg)

        target = {feature_name: batch[feature_name] for feature_name in self.ignore}
        features = {key: value for key, value in batch.items() if key not in self.ignore}

        sequentilal_features = [feature_name for feature_name, feature in features.items() if feature.dim() > 1]
        for feature_name in features:
            if feature_name in sequentilal_features:
                target[feature_name] = batch[feature_name][:, : -self.shift, ...].clone()
            else:
                target[feature_name] = batch[feature_name]

        target[self.out_feature_name] = batch[self.label_name][:, self.shift :, ...].clone()
        target[f"{self.out_feature_name}{self.mask_postfix}"] = batch[f"{self.label_name}{self.mask_postfix}"][
            :, self.shift :, ...
        ].clone()

        return target
