from abc import ABC, abstractmethod
from typing import Protocol

import torch

from replay.data.nn.schema import TensorMap


class AttentionMaskProto(Protocol):
    def __call__(self, feature_tensor: TensorMap, padding_mask: torch.BoolTensor) -> torch.Tensor: ...


class AttentionMaskBase(ABC):
    def __init__(
        self,
        num_heads: int,
    ) -> None:
        self.num_heads = num_heads

    def __call__(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """
        :param feature_tensor: dict of features tensors.
        :param padding_mask: Padding mask where ``0`` - ``<PAD>``, ``1`` - otherwise.
        :returns: Float attention mask of shape ``(B * num_heads, L, L)``,
            where ``-inf`` for ``<PAD>``, ``0`` - otherwise.
        """
        attention_mask = self._get_attention_mask(feature_tensor)

        diagonal_attention_mask = torch.diag(
            torch.ones(padding_mask.size(1), dtype=torch.bool, device=padding_mask.device)
        )
        # (B, L) -> (B, 1, 1, L)
        key_padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        # (B, 1, 1, L) -> (B, 1, L, L), where 0 - PAD, 1 - otherwise
        key_padding_mask = key_padding_mask | diagonal_attention_mask

        attention_mask = (attention_mask & key_padding_mask).float()
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf")).masked_fill(
            attention_mask == 1, 0.0
        )
        if attention_mask.size(1) != self.num_heads and attention_mask.shape[1] == 1:
            # for default attention_mask of shape (L, L) it becomes (B, 1, L, L)
            # (B, 1, L, L) -> (B, num_heads, L, L)
            attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)
        # (B, num_heads, L, L) -> (B * num_heads, L, L)
        attention_mask = attention_mask.reshape(-1, *attention_mask.shape[-2:])
        return attention_mask

    @abstractmethod
    def _get_attention_mask(self, feature_tensor: TensorMap) -> torch.Tensor:
        raise NotImplementedError()  # pragma: no cover


class DefaultAttentionMask(AttentionMaskBase):
    """
    Constructs a float lower-triangular attenstion mask
    of shape ``(batch_size * num_heads, sequence_length, sequence_length)``,
    where ``-inf`` for ``<PAD>``, ``0`` - otherwise.
    """

    def __init__(
        self,
        reference_feature_name: str,
        num_heads: int,
    ) -> None:
        """
        :param reference_feature_name: To build a mask, you need a reference tensor.
            So you need to pass the name of the tensor, which will definitely be in the dictionary of feature tensors.
            The second dimension (1 in zero indexing) of the tensor will be used to construct the attention mask.
        :param num_heads: Number of attention heads.
        """
        super().__init__(num_heads)
        self._feature_name = reference_feature_name

    def _get_attention_mask(self, feature_tensor: TensorMap) -> torch.Tensor:
        input_sequence = feature_tensor[self._feature_name]
        return torch.tril(
            torch.ones(
                (input_sequence.size(1), input_sequence.size(1)),
                dtype=torch.bool,
                device=input_sequence.device,
            )
        )
