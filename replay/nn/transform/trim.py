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
        feature_names: list[str] | str,
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

            output_batch[name] = output_batch[name][:, -self.seq_len :, ...].clone()

        return output_batch


class MaxBatchSeqlenTrimTransform(torch.nn.Module):
    """
    Trims sequences of specified names `feature_names` to the maximum sequence length in the batch.
    """

    def __init__(
        self,
        feature_names: list[str] | str,
        padding_mask_name: str = "padding_mask",
    ) -> None:
        """
        :param feature_name: name of features in batch to be trimmed.
        :param padding_mask_name: name of padding_mask in batch.
        """
        super().__init__()
        self.feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
        self.padding_mask_name = padding_mask_name

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.padding_mask_name not in batch:
            msg = f"Padding mask '{self.padding_mask_name}' not found in batch."
            raise KeyError(msg)

        source_seqlen = batch[self.padding_mask_name].size(1)
        max_non_padded_seqlen = batch[self.padding_mask_name].sum(dim=-1).max().item() + 1

        if max_non_padded_seqlen < source_seqlen:
            output_batch = dict(batch.items())

            for name in self.feature_names:
                output_batch[name] = (
                    output_batch[name]
                    .narrow(dim=1, start=-max_non_padded_seqlen, length=max_non_padded_seqlen)
                    .clone()
                    .contiguous()
                )
            return output_batch
        else:
            return batch
