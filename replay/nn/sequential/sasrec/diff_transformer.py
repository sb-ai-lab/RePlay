import math

import torch

from replay.data.nn import TensorMap
from replay.nn.attention import MultiHeadDifferentialAttention
from replay.nn.ffn import SwiGLU


class DiffTransformerBlock(torch.nn.Module):
    """
    Single Block of the DiffTransformer Architecture.
    Consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.

    Source paper: https://arxiv.org/pdf/2410.05258
    """

    def __init__(self, embedding_dim: int, num_heads: int, lambda_init: float):
        """
        :param embedding_dim: Total dimension of the model. Must be divisible by ``num_heads``.
        :param num_heads: Number of parallel attention heads.
        :param lambda_init: Initial value for lambda.
        """
        super().__init__()
        self.attn_norm = torch.nn.RMSNorm(embedding_dim)
        self.attn = MultiHeadDifferentialAttention(embedding_dim, num_heads, lambda_init, vdim=2 * embedding_dim)
        self.ff_norm = torch.nn.RMSNorm(embedding_dim)
        self.ff = SwiGLU(embedding_dim, 2 * embedding_dim)

    def reset_parameters(self) -> None:
        self.attn_norm.reset_parameters()
        self.attn.reset_parameters()
        self.ff_norm.reset_parameters()
        self.ff.reset_parameters()

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        Forward pass for a single differential transformer block.

        :param input_embeddings: Input tensor of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param attention_mask: Causal-like mask for attention pattern, where ``-inf`` for ``PAD``, ``0`` - otherwise.\n
            Possible shapes:\n
            1. ``(batch_size * num_heads, sequence_length, sequence_length)``
            2. ``(batch_size, num_heads, sequence_length, sequence_length)``
        :returns: Output tensor after processing through the block.
        """
        # Apply Multi-Head Differential Attention with residual connection
        attent_emb = self.attn(
            input_embeddings,
            input_embeddings,
            input_embeddings,
            attention_mask,
        )
        attention_block_out = self.attn_norm(attent_emb + input_embeddings)

        # Apply SwiGLU Feed-Forward Network with residual connection
        ff_out = self.ff(input_embeddings=attention_block_out)
        feedforward_block_out = self.ff_norm(ff_out + attention_block_out)
        return feedforward_block_out


class DiffTransformerLayer(torch.nn.Module):
    """
    Stacked blocks of the DiffTransformer Architecture.
    Single block consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.

    Source paper: https://arxiv.org/pdf/2410.05258\n
    Reference: https://github.com/nanowell/Differential-Transformer-PyTorch/blob/main/DiffTransformer.py
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_blocks: int,
    ) -> None:
        """
        :param embedding_dim: Total dimension of the model. Must be divisible by num_heads.
        :param num_heads: Number of parallel attention heads.
        :param num_blocks: Number of Transformer blocks.
        """
        torch.nn.MultiheadAttention
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                DiffTransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    lambda_init=0.8 - 0.6 * math.exp(-0.3 * block_num),
                )
                for block_num in range(num_blocks)
            ]
        )

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        feature_tensors: TensorMap,  # noqa: ARG002
        input_embeddings: torch.Tensor,
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        attention_mask: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        forward(input_embeddings, attention_mask)
        :param input_embeddings: Input tensor of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param attention_mask: Causal-like mask for attention pattern, where ``-inf`` for ``PAD``, ``0`` - otherwise.\n
            Possible shapes:\n
            1. ``(batch_size * num_heads, sequence_length, sequence_length)``
            2. ``(batch_size, num_heads, sequence_length, sequence_length)``
        :returns: Output tensor after processing through the layer.
        """
        seqs = input_embeddings
        for layer in self.layers:
            seqs = layer(
                input_embeddings=seqs,
                attention_mask=attention_mask,
            )
        return seqs
