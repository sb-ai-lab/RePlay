from typing import List, Union

import torch

from replay.data.nn.parquet.impl.masking import DEFAULT_MASK_POSTFIX
from replay.nn.transforms.base import BaseTransform


class NextTokenTransform(BaseTransform):
    """
    For the tensor specified by key ``label_field`` (typically "item_id") in the batch, this transform creates
    a corresponding "labels" tensor with a key ``out_feature_name`` in the batch, shifted forward
    by the specified ``shift`` value. This "labels" tensor are a target that model predicts.
    Padding mask for "labels" is also created. For all the other features excepted ``query_features``,
    last ``shift`` elements are truncated.

    This transform is required for the sequential models optimizing next token prediction task.

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
        >>> transform = NextTokenTransform(label_field="item_id", shift=1, query_features="user_id")
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'user_id': tensor([111]),
        'item_id': tensor([[5, 0, 7]]),
        'item_id_mask': tensor([[False,  True,  True]]),
        'labels': tensor([[0, 7, 4]]),
        'labels_mask': tensor([[True, True, True]])}

    """

    def __init__(
        self,
        label_field: str,
        shift: int = 1,
        query_features: Union[List[str], str] = "query_id",
        out_feature_name: str = "positive_labels",
        mask_postfix: str = DEFAULT_MASK_POSTFIX,
    ) -> None:
        """
        :param label_field: Name of target feature tensor to convert into labels.
        :param shift: Number of sequence items to shift by. Default: `1`.
        :param query_features: Name of the query column or list of user features.
            These columns will be excepted from the shifting and will be stayed unchanged. Default: ``"query_id"``.
        :param out_feature_name: The name of result feature in batch. Default: ``"positive_labels"``.
        :param mask_postfix: Postfix to append to the mask feature corresponding to resulting feature.
            Default: ``"_mask"``.
        """
        super().__init__()
        self.label_field = label_field
        self.shift = shift
        self.query_features = [query_features] if isinstance(query_features, str) else query_features
        self.out_feature_name = out_feature_name
        self.mask_postfix = mask_postfix

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if batch[self.label_field].dim() < 2:
            msg = (
                f"Transform expects batch feature {self.label_field} to be sequential "
                f"but tensor of shape {batch[self.label_field].shape} found."
            )
            raise ValueError(msg)

        max_len = batch[self.label_field].shape[1]
        if self.shift >= max_len:
            msg = (
                f"Transform with shift={self.shift} cannot be applied to sequences of length {max_len}."
                "Decrease value of `shift` parameter in transform"
            )
            raise ValueError(msg)

        target = {feature_name: batch[feature_name] for feature_name in self.query_features}
        features = {key: value for key, value in batch.items() if key not in self.query_features}

        sequentilal_features = [feature_name for feature_name, feature in features.items() if feature.dim() > 1]
        for feature_name in features:
            if feature_name in sequentilal_features:
                target[feature_name] = batch[feature_name][:, : -self.shift, ...].clone()
            else:
                target[feature_name] = batch[feature_name]

        target[self.out_feature_name] = batch[self.label_field][:, self.shift :, ...].clone()
        target[f"{self.out_feature_name}{self.mask_postfix}"] = batch[f"{self.label_field}{self.mask_postfix}"][
            :, self.shift :, ...
        ].clone()

        return target
