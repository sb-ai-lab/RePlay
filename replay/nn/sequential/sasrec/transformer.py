import contextlib
from typing import Literal

import torch

from replay.data.nn import TensorMap
from replay.nn.ffn import PointWiseFeedForward


class SasRecTransformerLayer(torch.nn.Module):
    """
    SasRec vanilla layer.
    Layer consists of Multi-Head Attention followed by a Point-Wise Feed-Forward Network.

    Source paper: https://arxiv.org/pdf/1808.09781.pdf
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
        activation: Literal["relu", "gelu"] = "gelu",
    ) -> None:
        """
        :param embedding_dim: Total dimension of the model. Must be divisible by num_heads.
        :param num_heads: Number of parallel attention heads.
        :param num_blocks: Number of Transformer blocks.
        :param dropout: probability of an element to be zeroed.
        :param activation: the name of the activation function.
            Default: ``"gelu"``.
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.attention_layers = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.attention_layernorms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(embedding_dim, eps=1e-8) for _ in range(num_blocks)]
        )
        self.forward_layers = torch.nn.ModuleList(
            [
                PointWiseFeedForward(
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_blocks)
            ]
        )
        self.forward_layernorms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(embedding_dim, eps=1e-8) for _ in range(num_blocks)]
        )

    def reset_parameters(self):
        for i in range(self.num_blocks):
            self.attention_layernorms[i].reset_parameters()
            self.forward_layernorms[i].reset_parameters()
            self.forward_layers[i].reset_parameters()

        for _, param in self.attention_layers.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(
        self,
        feature_tensors: TensorMap,  # noqa: ARG002
        input_embeddings: torch.Tensor,
        padding_mask: torch.BoolTensor,
        attention_mask: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        :param input_embeddings: Input tensor of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param padding_mask: A mask of shape ``(batch_size, sequence_length)`` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding").
            ``False`` value indicates that the corresponding ``key`` value will be ignored.
        :param attention_mask: Causal-like mask for attention pattern, where ``-inf`` for ``PAD``, ``0`` - otherwise.\n
            Possible shapes:\n
            1. ``(batch_size * num_heads, sequence_length, sequence_length)``\n
            2. ``(batch_size, num_heads, sequence_length, sequence_length)``
        :returns: torch.Tensor: Output tensor after processing through the layer.
        """
        seqs = input_embeddings
        key_padding_mask = torch.zeros_like(padding_mask, dtype=torch.float32).masked_fill_(
            padding_mask.logical_not(), -torch.inf if self.training else torch.finfo(torch.float32).min
        )
        for i in range(self.num_blocks):
            query = self.attention_layernorms[i](seqs)
            attn_emb, _ = self.attention_layers[i](
                query,
                seqs,
                seqs,
                attn_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            seqs = query + attn_emb
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
        return seqs
