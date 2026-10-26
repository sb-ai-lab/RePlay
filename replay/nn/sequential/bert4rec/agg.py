import contextlib

import torch

from replay.data.nn.schema import TensorMap
from replay.nn.agg import AggregatorProto


class BERTPositionAwareAggregator(torch.nn.Module):
    """
    Aggregates feature embeddings, applies BERT4Rec token masking, and adds positional embeddings.
    """

    def __init__(
        self,
        embedding_aggregator: AggregatorProto,
        max_sequence_length: int,
        dropout: float,
    ) -> None:
        """
        :param embedding_aggregator: An object of a class that performs the logic of aggregating multiple embeddings.\n
            For example, it can be a ``sum``, a ``mean``, or a ``concatenation``.
        :param max_sequence_length: Maximum sequence length for learned positional embeddings.
        :param dropout: Probability of an element to be zeroed.
        """
        super().__init__()
        self.embedding_aggregator = embedding_aggregator
        self.mask_embedding = torch.nn.Embedding(1, self.embedding_aggregator.embedding_dim)
        self.pe = torch.nn.Embedding(max_sequence_length, self.embedding_aggregator.embedding_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def reset_parameters(self) -> None:
        self.embedding_aggregator.reset_parameters()
        for _, param in self.pe.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)
        for _, param in self.mask_embedding.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(self, feature_tensors: TensorMap, token_mask: torch.BoolTensor) -> torch.Tensor:
        """
        :param feature_tensors: a dictionary of tensors to pass into ``embedding_aggregator``.
        :param token_mask: Token mask where ``False`` marks positions to replace with the mask embedding.

        :returns: Aggregated embeddings with positional encoding.
        """
        seqs: torch.Tensor = self.embedding_aggregator(feature_tensors)
        assert seqs.dim() == 3
        batch_size, seq_len, embedding_dim = seqs.size()
        assert seq_len <= self.pe.num_embeddings, (
            f"Sequence length = {seq_len} is greater than positional embedding num = {self.pe.num_embeddings}"
        )

        expanded_mask = token_mask.unsqueeze(-1).expand(-1, -1, embedding_dim)
        expanded_embedding = self.mask_embedding.weight.expand(batch_size, seq_len, -1)
        seqs = seqs.where(expanded_mask, expanded_embedding)

        seqs += self.pe.weight[-seq_len:].unsqueeze(0).repeat(batch_size, 1, 1)
        seqs = self.dropout(seqs)
        return seqs
