from typing import List, Union

import torch

from replay.nn.transforms.base import BaseTransform


class TrimTransform(BaseTransform):
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

            trimmed_seq = output_batch[name][:, -self.seq_len :, ...].clone()
            output_batch[name] = trimmed_seq

        return output_batch
