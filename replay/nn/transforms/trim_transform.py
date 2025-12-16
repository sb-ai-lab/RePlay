from typing import List, Union

import torch

from replay.nn.transforms.base import BaseTransform


class TrimTransform(BaseTransform):
    """
    Trims sequences of specified names `feature_names` keeping the specified sequence length `seq_len` on the right.

    Example:

    .. code-block:: python

        >>> input_batch = {
        >>>     "user_id": torch.LongTensor([111]),
        >>>     "item_id": torch.LongTensor([[5, 0, 7, 4]]),
        >>>     "seen_ids": torch.BoolTensor([[5, 0, 7, 4]])}
        >>> transform = TrimTransform(seq_len=3, feature_name="item_id")
        >>> output_batch = transform(input_batch)
        >>> output_batch
        {'user_id': tensor([111]),
         'item_id': tensor([[5, 0, 7]]),
         'seen_ids': tensor([[5, 0, 7, 4]]),}

    """

    def __init__(
        self,
        seq_len: int,
        feature_names: Union[List[str], str] = "query_id",
    ) -> None:
        """
        :param seq_len: max sequence length used in model.
        :param feature_name: name of feature in batch to be trimmed
        """
        super().__init__()
        self.seq_len = seq_len
        self.feature_names = [feature_names] if isinstance(feature_names, str) else feature_names

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name in self.feature_names:
            assert batch[name].shape[1] >= self.seq_len

            trimmed_seq = batch[name][:, -self.seq_len :, ...].clone()
            batch[name] = trimmed_seq

        return batch
