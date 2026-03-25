import torch


class MaxBatchSeqlenTrimTransform(torch.nn.Module):
    """
    Trims sequences of specified names `feature_names` to the maximum sequence_length in the batch.
    """

    def __init__(
        self,
        feature_names: list[str] | str,
        padding_mask_name: str,
    ) -> None:
        """
        :param feature_name: name of feature in batch to be trimmed to the max_batch_seq_len
        :param padding_mask_name: name of padding_mask
        """
        super().__init__()
        self.feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
        self.padding_mask_name = padding_mask_name

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        source_seq_len = batch[self.padding_mask_name].size(1)
        max_batch_seq_len = min(source_seq_len, batch[self.padding_mask_name].sum(dim=-1).max() + 1)  # `+1` - for train, when preidcting by the last padding token - first item in a sequence (first target_padding_mask value).
        if source_seq_len == max_batch_seq_len:
            return batch

        for name in self.feature_names:
            batch[name] = torch.narrow(batch[name], 1, -max_batch_seq_len, max_batch_seq_len).contiguous()  # w/o contigous got error on categorical embeddings during indices.view(...)
        return batch
