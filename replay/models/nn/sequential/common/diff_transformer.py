import contextlib
import math
from typing import Optional

import torch
import torch.nn.functional as f

from replay.data.nn import TensorMap

from .ffn import SwiGLU


class MultiHeadDifferentialAttention(torch.nn.Module):
    """
    Multi-Head Differential Attention Mechanism.
    Replaces the conventional softmax attention with a differential attention.
    Incorporates a causal mask (if other not specified) to ensure autoregressive behavior.

    Paper: https://arxiv.org/pdf/2410.05258
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        lambda_init: float,
        bias: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):
        """
        :param embedding_dim: Total dimension of the model. Must be divisible by ``num_heads``.
        :param num_heads: Number of parallel attention heads.
        :param lambda_init: Initial value for lambda.
        :param bias: If specified, adds bias to input / output projection layers. Default: ``False``.
        :param kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embedding_dim``).
        :param vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embedding_dim``).
        """
        super().__init__()
        kdim = kdim or embedding_dim
        vdim = vdim or embedding_dim
        assert kdim % num_heads == 0, "Query/Key embedding dim is not divisible by num_heads"
        assert vdim % num_heads == 0, "Value embedding dim is not divisible by num_heads"
        self.qk_e_head = kdim // num_heads
        self.v_e_head = vdim // num_heads
        self.num_heads = num_heads

        # Linear projections for queries, keys, and values
        # Project to 2 * d_head per head for differential attention
        self.W_q = torch.nn.Linear(embedding_dim, 2 * self.qk_e_head * num_heads, bias=bias)
        self.W_k = torch.nn.Linear(embedding_dim, 2 * self.qk_e_head * num_heads, bias=bias)
        self.W_v = torch.nn.Linear(embedding_dim, self.v_e_head * num_heads, bias=bias)
        self.W_o = torch.nn.Linear(self.v_e_head * num_heads, embedding_dim, bias=bias)

        # Learnable parameters for lambda reparameterization
        self.lambda_q1 = torch.nn.Parameter(torch.randn(num_heads, self.qk_e_head))
        self.lambda_k1 = torch.nn.Parameter(torch.randn(num_heads, self.qk_e_head))
        self.lambda_q2 = torch.nn.Parameter(torch.randn(num_heads, self.qk_e_head))
        self.lambda_k2 = torch.nn.Parameter(torch.randn(num_heads, self.qk_e_head))
        self.register_buffer("scaling", torch.asarray(1 / math.sqrt(self.qk_e_head), dtype=torch.float32))

        self.lambda_init = lambda_init

        # Scale parameter for RMSNorm
        self.rms_scale = torch.nn.Parameter(torch.ones(self.v_e_head))
        self.eps = 1e-5  # Epsilon for numerical stability

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        Forward pass for Multi-Head Differential Attention.

        :param query: Query sequence of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param key: Key sequence of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param value: Value sequence of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param attn_mask: attention mask, where ``-inf`` for ``PAD``, ``0`` - otherwise.\n
            Possible shapes:\n
            1. ``(batch_size * num_heads, sequence_length, sequence_length)``
            2. ``(batch_size, num_heads, sequence_length, sequence_length)``
        :returns: torch.Tensor: Output tensor after applying differential attention.
        """
        batch_size, seq_len, _ = value.shape

        # Project inputs to queries, keys, and values
        query = self.W_q(query)  # Shape: (batch_size, seq_len, 2 * num_heads * qk_e_head)
        key = self.W_k(key)  # Shape: (batch_size, seq_len, 2 * num_heads * qk_e_head)
        value = self.W_v(value)  # Shape: (batch_size, seq_len, num_heads * v_e_head)

        # Reshape and permute for multi-head attention
        # New shape: (batch_size, num_heads, sequence_length, 2 * qk_e_head or v_e_head)
        query = query.view(batch_size, seq_len, self.num_heads, 2 * self.qk_e_head).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, 2 * self.qk_e_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.v_e_head).transpose(1, 2)

        # Split query and key into query1, query2 and key1, key2
        query1, query2 = query.chunk(2, dim=-1)  # Each of shape: (batch_size, num_heads, seq_len, d_head)
        key1, key2 = key.chunk(2, dim=-1)  # Each of shape: (batch_size, num_heads, seq_len, d_head)

        # Compute lambda using reparameterization
        # Compute dot products for each head
        # Shape of lambda_val: (num_heads,)
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()  # (num_heads,)
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()  # (num_heads,)
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init  # (num_heads,)

        # Expand lambda_val to match attention dimensions (batch_size, num_heads, 1, 1)
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # Reshape attn_mask from 3D  to 4D
        if len(attn_mask.shape) == 3:
            attn_mask = attn_mask.reshape(attn_mask.shape[0] // self.num_heads, self.num_heads, *attn_mask.shape[1:])

        # check shapes
        assert attn_mask.dim() == 4
        assert attn_mask.size() == (batch_size, self.num_heads, seq_len, seq_len)

        # Compute attention scores
        attention_scores1 = torch.matmul(query1, key1.transpose(-2, -1)) * self.get_buffer(
            "scaling"
        )  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores2 = torch.matmul(query2, key2.transpose(-2, -1)) * self.get_buffer(
            "scaling"
        )  # (batch_size, num_heads, seq_len, seq_len)

        # Apply the causal mask
        attention_scores1 = attention_scores1 + attn_mask  # Mask out future positions
        attention_scores2 = attention_scores2 + attn_mask  # Mask out future positions

        # Apply softmax to get attention weights
        attention1 = f.softmax(attention_scores1, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention2 = f.softmax(attention_scores2, dim=-1)
        attention = attention1 - lambda_val * attention2

        # Apply attention weights to values
        output = torch.matmul(attention, value)  # (batch_size, num_heads, seq_len, v_e_head)

        # Normalize each head independently using RMSNorm
        # First, reshape for RMSNorm
        output_reshaped = output.contiguous().view(
            batch_size * self.num_heads,
            seq_len,
            self.v_e_head,
        )  # (batch_size*num_heads, seq_len, v_e_head)

        # Compute RMSNorm
        rms_norm = torch.sqrt(
            output_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )  # (batch_size*num_heads, seq_len, 1)
        output_normalized = (output_reshaped / rms_norm) * self.rms_scale  # (batch*num_heads, seq_len, v_e_head)

        # Reshape back to (batch_size, num_heads, seq_len, v_e_head)
        output_normalized = output_normalized.view(batch_size, self.num_heads, seq_len, self.v_e_head)

        # Scale the normalized output
        output_normalized = output_normalized * (1 - self.lambda_init)  # Scalar scaling

        # Concatenate all heads
        # New shape: (batch_size, seq_len, num_heads * v_e_head)
        output_concat = (
            output_normalized.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.v_e_head)
        )

        # Final linear projection
        output_projection = self.W_o(output_concat)  # (batch_size, seq_len, embedding_dim)
        return output_projection


class DiffTransformerBlock(torch.nn.Module):
    """
    Single Block of the DiffTransformer Architecture.
    Consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.

    Paper: https://arxiv.org/pdf/2410.05258
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

    Paper: https://arxiv.org/pdf/2410.05258\n
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
                DiffTransformerLayer(
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
