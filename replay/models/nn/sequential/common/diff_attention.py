import contextlib
import math
from typing import Optional

import torch
import torch.nn.functional as f

from replay.data.nn import TensorMap

from .ffn import SwiGLU
from .normalization import RMSNorm


class MultiHeadDifferentialAttention(torch.nn.Module):
    """
    Multi-Head Differential Attention Mechanism.
    Replaces the conventional softmax attention with a differential attention.
    Incorporates a causal mask (if other not specified) to ensure autoregressive behavior.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        lambda_init: float,
        bias: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):
        """
        Args:
            hidden_size (int): Dimension of the model. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda.
            bias (bool): whether to add bias to linear projections for query, key, value and output projection
            kdim (int): query/key output projection dimensions
            vdim (int): value output projection dimension
        """
        super().__init__()
        kdim = kdim or hidden_size
        vdim = vdim or hidden_size
        assert kdim % num_heads == 0, "Query/Key embedding dim is not divisible by num_heads"
        assert vdim % num_heads == 0, "Value embedding dim is not divisible by num_heads"
        self.qk_e_head = kdim // num_heads
        self.v_e_head = vdim // num_heads
        self.num_heads = num_heads

        # Linear projections for queries, keys, and values
        # Project to 2 * d_head per head for differential attention
        self.W_q = torch.nn.Linear(hidden_size, 2 * self.qk_e_head * num_heads, bias=bias)
        self.W_k = torch.nn.Linear(hidden_size, 2 * self.qk_e_head * num_heads, bias=bias)
        self.W_v = torch.nn.Linear(hidden_size, self.v_e_head * num_heads, bias=bias)
        self.W_o = torch.nn.Linear(self.v_e_head * num_heads, hidden_size, bias=bias)

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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Multi-Head Differential Attention.

        Args:
            query (Tensor): Query sequence of shape (batch, sequence_length, hidden_size).
            key (Tensor): Key sequence of shape (batch, sequence_length, hidden_size).
            value (Tensor): Value sequence of shape (batch, sequence_length, hidden_size).
            attn_mask FloatTensor: attention mask, where -inf for PAD, 0 otherwise.
            Possible shapes:
             - (batch * num_heads, sequence_length, sequence_length)
             - (batch, num_heads, sequence_length, sequence_length)
        Returns:
            Tensor: Output tensor after applying differential attention.
        """
        batch, seq_len, _ = value.shape

        # Project inputs to queries, keys, and values
        query = self.W_q(query)  # Shape: (batch, seq_len, 2 * num_heads * qk_e_head)
        key = self.W_k(key)  # Shape: (batch, seq_len, 2 * num_heads * qk_e_head)
        value = self.W_v(value)  # Shape: (batch, seq_len, num_heads * v_e_head)

        # Reshape and permute for multi-head attention
        # New shape: (batch, num_heads, sequence_length, 2 * qk_e_head or v_e_head)
        query = query.view(batch, seq_len, self.num_heads, 2 * self.qk_e_head).transpose(1, 2)
        key = key.view(batch, seq_len, self.num_heads, 2 * self.qk_e_head).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_heads, self.v_e_head).transpose(1, 2)

        # Split query and key into query1, query2 and key1, key2
        query1, query2 = query.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, seq_len, d_head)
        key1, key2 = key.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, seq_len, d_head)

        # Compute lambda using reparameterization
        # Compute dot products for each head
        # Shape of lambda_val: (num_heads,)
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()  # (num_heads,)
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()  # (num_heads,)
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init  # (num_heads,)

        # Expand lambda_val to match attention dimensions (batch, num_heads, 1, 1)
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # Здесь маску batch_size*num_heads, seq_len, seq_len делаем 4d
        if len(attn_mask.shape) == 3:
            attn_mask = attn_mask.reshape(attn_mask.shape[0] // self.num_heads, self.num_heads, *attn_mask.shape[1:])

        # Убеждаемся что подается верная 4d маска
        if len(attn_mask.shape) != 4 or attn_mask.shape[1] != self.num_heads:
            msg = (
                "Expected to get attention_mask of shape (batch_size*num_heads, seq_len, seq_len) "
                f"But get attention_mask shape: {attn_mask.shape}"
            )
            raise ValueError(msg)

        # Compute attention scores
        attention_scores1 = torch.matmul(query1, key1.transpose(-2, -1)) * self.get_buffer(
            "scaling"
        )  # (batch, num_heads, seq_len, seq_len)
        attention_scores2 = torch.matmul(query2, key2.transpose(-2, -1)) * self.get_buffer(
            "scaling"
        )  # (batch, num_heads, seq_len, seq_len)

        # Apply the causal mask
        attention_scores1 = attention_scores1 + attn_mask  # Mask out future positions
        attention_scores2 = attention_scores2 + attn_mask  # Mask out future positions

        # Apply softmax to get attention weights
        attention1 = f.softmax(attention_scores1, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        attention2 = f.softmax(attention_scores2, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        attention = attention1 - lambda_val * attention2  # (batch, num_heads, seq_len, seq_len)

        # Apply attention weights to values
        output = torch.matmul(attention, value)  # (batch, num_heads, seq_len, v_e_head)

        # Normalize each head independently using RMSNorm
        # First, reshape for RMSNorm
        output_reshaped = output.contiguous().view(
            batch * self.num_heads, seq_len, self.v_e_head
        )  # (batch*num_heads, seq_len, v_e_head)

        # Compute RMSNorm
        rms_norm = torch.sqrt(
            output_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )  # (batch*num_heads, seq_len, 1)
        output_normalized = (output_reshaped / rms_norm) * self.rms_scale  # (batch*num_heads, seq_len, v_e_head)

        # Reshape back to (batch, num_heads, seq_len, v_e_head)
        output_normalized = output_normalized.view(batch, self.num_heads, seq_len, self.v_e_head)

        # Scale the normalized output
        output_normalized = output_normalized * (1 - self.lambda_init)  # Scalar scaling

        # Concatenate all heads
        # New shape: (batch, seq_len, num_heads * v_e_head)
        output_concat = (
            output_normalized.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.v_e_head)
        )

        # Final linear projection
        output_projection = self.W_o(output_concat)  # (batch, seq_len, hidden_size)
        return output_projection, None


class DiffTransformerLayer(torch.nn.Module):
    """
    Single Layer of the DiffTransformer Architecture.
    Consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.
    """

    def __init__(self, hidden_size: int, num_heads: int, lambda_init: float):
        """
        Args:
            hidden_size (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda in Differential Attention.
        """
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = MultiHeadDifferentialAttention(hidden_size, num_heads, lambda_init, vdim=2 * hidden_size)
        self.ff_norm = RMSNorm(hidden_size)
        self.ff = SwiGLU(hidden_size)

    def reset_parameters(self) -> None:
        self.attn_norm.reset_parameters()
        self.attn.reset_parameters()
        self.ff_norm.reset_parameters()
        self.ff.reset_parameters()

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.FloatTensor,
        feature_tensors: TensorMap,
    ) -> torch.Tensor:
        """
        Forward pass for a single transformer layer.

        Args:
            input_embeddings (Tensor): Input tensor of shape (batch, sequence_length, hidden_size).
            attention_mask (FloatTensor): Causal-like mask for attention pattern.
                Possible shapes:
                - (batch * num_heads, sequence_length, sequence_length)
                - (batch, num_heads, sequence_length, sequence_length)
        Returns:
            Tensor: Output tensor after processing through the layer.
        """
        # Apply Multi-Head Differential Attention with residual connection
        attent_emb, _ = self.attn(
            input_embeddings,
            input_embeddings,
            input_embeddings,
            attention_mask,
        )
        attention_block_out = self.attn_norm(attent_emb + input_embeddings)

        # Apply SwiGLU Feed-Forward Network with residual connection
        ff_out = self.ff(
            feature_tensors=feature_tensors,
            input_embeddings=attention_block_out,
        )
        feedforward_block_out = self.ff_norm(ff_out + attention_block_out)
        return feedforward_block_out


class DiffTransformerBlock(torch.nn.Module):
    """
    Stacked layers of the DiffTransformer Architecture.
    Single block consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.

    Paper: https://arxiv.org/pdf/2410.05258
    Code:  https://github.com/nanowell/Differential-Transformer-PyTorch/blob/main/DiffTransformer.py
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_blocks: int,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                DiffTransformerLayer(
                    hidden_size=hidden_size,
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
        feature_tensors: TensorMap,
        input_embeddings: torch.Tensor,
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        seqs = input_embeddings
        for layer in self.layers:
            seqs = layer(
                feature_tensors=feature_tensors,
                input_embeddings=seqs,
                attention_mask=attention_mask,
            )
        return seqs
