import contextlib
from typing import Any, Literal

import torch

from replay.data.nn import TensorMap
from replay.models.nn.sequential.common.ffn import PointWiseFeedForward


class SasRecBlock(torch.nn.Module):
    """
    SasRec vanilla layers:
        1. SelfAttention layers
        2. FeedForward layers

    Link: https://arxiv.org/pdf/1808.09781.pdf
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
        activation: Literal["relu", "gelu"] = "gelu",
    ) -> None:
        """
        :param hidden_size: Hidden size of transformer.
        :param num_heads: Number of Attention heads.
        :param num_blocks: Number of Transformer blocks.
        :param dropout: Dropout rate.
        :param activation: Non-linear activation function
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.attention_layers = self._layers_stacker(
            num_blocks,
            torch.nn.MultiheadAttention,
            hidden_size,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.attention_layernorms = self._layers_stacker(num_blocks, torch.nn.LayerNorm, hidden_size, eps=1e-8)
        self.forward_layers = self._layers_stacker(
            num_blocks,
            PointWiseFeedForward,
            hidden_size,
            dropout,
            activation,
        )
        self.forward_layernorms = self._layers_stacker(num_blocks, torch.nn.LayerNorm, hidden_size, eps=1e-8)

    def reset_parameters(self):
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(
        self,
        feature_tensors: TensorMap,  # noqa: ARG002
        input_embeddings: torch.Tensor,
        attention_mask: torch.BoolTensor,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        seqs = input_embeddings

        for i in range(self.num_blocks):
            query = self.attention_layernorms[i](seqs)
            attn_emb, _ = self.attention_layers[i](
                query,
                seqs,
                seqs,
                attn_mask=attention_mask,
                key_padding_mask=padding_mask.logical_not(),
                need_weights=False,
            )
            seqs = query + attn_emb
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
        return seqs

    def _layers_stacker(
        self,
        num_blocks: int,
        layer_class: Any,
        *args,
        **kwargs,
    ) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([layer_class(*args, **kwargs) for _ in range(num_blocks)])
