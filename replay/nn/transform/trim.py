from typing import List, Union

import torch


class TrimTransform(torch.nn.Module):
    """
    Trims sequences of specified names `feature_names` keeping the specified sequence length `seq_len` on the right.

    Example:

    .. code-block:: python

        >>> input_batch = {
        ...     "user_id": torch.LongTensor([111]),
        ...     "item_id": torch.LongTensor([[5, 4, 0, 7, 4]]),
        ...     "seen_ids": torch.LongTensor([[5, 4, 0, 7, 4]]),
        ... }
        >>> transform = TrimTransform(seq_len=3, feature_names="item_id")
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'user_id': tensor([111]),
        'item_id': tensor([[0, 7, 4]]),
        'seen_ids': tensor([[5, 4, 0, 7, 4]])}

    """

    def __init__(
        self,
        seq_len: int,
        feature_names: Union[List[str], str],
    ) -> None:
        """
        :param seq_len: max sequence length used in model. Must be positive.
        :param feature_name: name of feature in batch to be trimmed.
        """
        super().__init__()
        assert seq_len > 0
        self.seq_len = seq_len
        self.feature_names = [feature_names] if isinstance(feature_names, str) else feature_names

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_batch = dict(batch.items())

        for name in self.feature_names:
            assert output_batch[name].shape[1] >= self.seq_len
            output_batch[name] = output_batch[name][:, -self.seq_len :, ...].contiguous()

        return output_batch


class AdaptiveTrimTransform(torch.nn.Module):
    """
    Trims sequences of specified names `feature_names` to the maximum sequence length in the current batch.
    This transform is assumed to be used for validation and inference for speeding up due to reducing
    length of padded parts of sequences. Note that sequences should be left-padded.

    Example:

    .. code-block:: python

        >>> input_batch = {
        ...     "item_id": torch.LongTensor([[5, 5, 5, 5, 0], [5, 5, 0, 2, 4]]),
        ...     "padding_mask": torch.BoolTensor([[0, 0, 0, 0, 1], [0, 0, 1, 1, 1]]),
        ... }
        >>> transform = AdaptiveTrimTransform("item_id", padding_mask_name="padding_mask")
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'item_id': tensor([[5, 5, 0],
                 [0, 2, 4]]),
         'padding_mask': tensor([[False, False,  True],
                 [ True,  True,  True]])}


    """

    def __init__(
        self,
        feature_names: Union[List[str], str],
        padding_mask_name: str = "padding_mask",
    ) -> None:
        """
        :param feature_name: name of features in batch to be trimmed.
            `padding_mask_name` will be included automatically.
        :param padding_mask_name: name of padding_mask in batch.
        """
        super().__init__()
        self.feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
        if padding_mask_name not in self.feature_names:
            self.feature_names.append(padding_mask_name)

        self.padding_mask_name = padding_mask_name

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.padding_mask_name not in batch:
            msg = f"Padding mask '{self.padding_mask_name}' not found in batch."
            raise KeyError(msg)

        assert batch[self.padding_mask_name].ndim == 2
        source_seqlen = batch[self.padding_mask_name].size(1)
        max_non_padded_seqlen = batch[self.padding_mask_name].sum(dim=1).max().item()

        if max_non_padded_seqlen == source_seqlen:
            return batch

        output_batch = dict(batch.items())
        for name in self.feature_names:
            output_batch[name] = output_batch[name][:, -max_non_padded_seqlen:, ...].contiguous()
        return output_batch
