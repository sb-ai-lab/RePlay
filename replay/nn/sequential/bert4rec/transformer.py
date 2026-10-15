import contextlib
from typing import Literal

import torch

from replay.data.nn import TensorMap
from replay.nn.ffn import PositionwiseFeedForward


class BERT4RecTransformerBlock(torch.nn.Module):
    """
    BERT4Rec transformer block with bidirectional self-attention and a feed-forward network.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_hidden: int,
        dropout: float,
        activation: Literal["relu", "gelu"] = "gelu",
    ) -> None:
        """
        :param embedding_dim: Hidden size of transformer.
        :param num_heads: Number of attention heads.
        :param feed_forward_hidden: Hidden size of the feed-forward network, usually ``4 * embedding_dim``.
        :param dropout: Dropout probability.
        :param activation: Activation function used in the feed-forward network.
        """
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.attention_norm = torch.nn.LayerNorm(embedding_dim)

        self.pff = PositionwiseFeedForward(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=feed_forward_hidden,
            dropout=dropout,
            activation=activation,
        )
        self.pff_dropout = torch.nn.Dropout(dropout)
        self.pff_norm = torch.nn.LayerNorm(embedding_dim)

        self.dropout = torch.nn.Dropout(p=dropout)

    def reset_parameters(self) -> None:
        self.attention_norm.reset_parameters()
        self.pff.reset_parameters()
        self.pff_norm.reset_parameters()
        for _, param in self.attention.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        :param input_embeddings: Input sequence embeddings.
        :param padding_mask: Padding mask where ``False`` marks padding positions.

        :returns: Embeddings after transformer block is applied.
        """
        # Attention + skip-connection
        x_norm = self.attention_norm(input_embeddings)
        attent_emb, _ = self.attention(
            x_norm, x_norm, x_norm, key_padding_mask=padding_mask.logical_not(), need_weights=False
        )
        y = input_embeddings + self.attention_dropout(attent_emb)

        # PFF + skip-connection
        z = y + self.pff_dropout(self.pff(self.pff_norm(y)))

        return self.dropout(z)


class Bert4RecTransformerLayer(torch.nn.Module):
    """
    Stack of BERT4Rec transformer blocks.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        feed_forward_hidden: int,
        dropout: float,
        activation: Literal["relu", "gelu"] = "gelu",
    ) -> None:
        """
        :param embedding_dim: Hidden size of transformer.
        :param num_heads: Number of attention heads.
        :param num_blocks: Number of transformer blocks.
        :param feed_forward_hidden: Hidden size of the feed-forward network, usually ``4 * embedding_dim``.
        :param dropout: Dropout probability.
        :param activation: Activation function used in the feed-forward network.
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BERT4RecTransformerBlock(
                    embedding_dim,
                    num_heads,
                    feed_forward_hidden,
                    dropout,
                    activation,
                )
                for _ in range(num_blocks)
            ]
        )

    def reset_parameters(self) -> None:
        for i in range(self.num_blocks):
            self.transformer_blocks[i].reset_parameters()

    def forward(
        self,
        feature_tensors: TensorMap,  # noqa: ARG002
        input_embeddings: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        :param feature_tensors: Input feature tensors. Reserved for encoders that use feature-aware logic.
        :param input_embeddings: Input sequence embeddings.
        :param padding_mask: Padding mask where ``False`` marks padding positions.

        :returns: Embeddings after all transformer blocks are applied.
        """
        seqs = input_embeddings
        for transformer in self.transformer_blocks:
            seqs = transformer(seqs, padding_mask)

        return seqs
