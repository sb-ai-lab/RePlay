import contextlib
from typing import Optional, Protocol

import torch

from replay.data.nn import TensorMap, TensorSchema


class SasRecEmbeddingProtocol(Protocol):
    def get_item_weights(self, indices: Optional[torch.LongTensor] = None) -> torch.Tensor: ...

    def get_all_embeddings(self) -> TensorMap: ...

    def reset_parameters(self) -> None: ...

class SasRecPositionalEmbedding(torch.nn.Module):
    """
    Positional embedding.
    """

    def __init__(self, max_len: int, embedding_dim: int) -> None:
        """
        :param max_len: Max sequence length.
        :param embedding_dim: Embedding dimension.
        """
        super().__init__()
        self.pe = torch.nn.Embedding(max_len, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input embedding.

        :returns: Positional embedding.
        """
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class SasRecEmbedding(torch.nn.Module, SasRecEmbeddingProtocol):
    """
    SasRec Embedding:
        1. ItemEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information

    Link: https://arxiv.org/pdf/1808.09781.pdf
    """

    def __init__(
        self,
        schema: TensorSchema,
        max_len: int,
        dropout: float,
    ) -> None:
        """
        :param schema Tensor schema of features.
        :param max_len: Max length of sequence.
        :param dropout: Dropout rate.
        """
        super().__init__()
        assert schema.item_id_feature_name
        self.item_feature_name = schema.item_id_feature_name

        item_count = schema.item_id_features.item().cardinality
        padding_idx = schema.item_id_features.item().padding_value
        embedding_dim = schema.item_id_features.item().embedding_dim

        self.item_emb = torch.nn.Embedding(item_count, embedding_dim, padding_idx=padding_idx)
        self.pos_emb = SasRecPositionalEmbedding(max_len=max_len, embedding_dim=embedding_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

        if self.item_emb.padding_idx is not None:
            self.item_emb.weight.data[self.item_emb.padding_idx] = 0


    def forward(self, feature_tensor: TensorMap) -> torch.Tensor:
        """
        :param feature_tensor: Batch of features.

        :returns: Embeddings for input features.
        """
        seqs = self.item_emb(feature_tensor[self.item_feature_name]) * (self.item_emb.embedding_dim**0.5)
        seqs += self.pos_emb(seqs)
        seqs = self.dropout(seqs)
        return seqs

    def get_item_weights(self, indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        :param indices: Items indices.
            Default: `None`, means return weights for all items.

        :returns: Item weights for specific items.
        """
        if indices is None:
            return self.item_emb.weight

        return self.item_emb(indices)

    def get_all_embeddings(self) -> TensorMap:
        """
        :returns: copy of all embeddings presented in this layer as a dict.
        """
        return {
            "item_embedding": self.item_emb.weight.data.detach().clone(),
            "positional_embedding": self.pos_emb.pe.weight.data.detach().clone(),
        }
