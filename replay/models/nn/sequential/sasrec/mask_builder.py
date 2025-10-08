from typing import Protocol

import torch

from replay.data.nn import TensorMap, TensorSchema


class SasRecAttentionMaskBuilderProtocol(Protocol):
    def __call__(self, feature_tensor: TensorMap, padding_mask: torch.BoolTensor) -> torch.Tensor: ...

class SasRecAttentionMaskBuilder(SasRecAttentionMaskBuilderProtocol):
    def __init__(self, schema: TensorSchema) -> None:
        assert schema.item_id_feature_name
        self.item_feature_name = schema.item_id_feature_name

    def __call__(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        input_sequence = feature_tensor[self.item_feature_name]
        seq_len = input_sequence.size(1)

        attention_mask = ~torch.tril(torch.ones(
            size=(seq_len, seq_len),
            dtype=torch.bool,
            device=padding_mask.device,
            requires_grad=False,
        ))
        return attention_mask
