import contextlib

import torch

from replay.data.nn.schema import TensorMap
from replay.models.nn.sequential.common.agg import SequentialEmbeddingAggregatorProto


class QueryTowerEmbeddingAggregator(torch.nn.Module):
    def __init__(
        self,
        embedding_aggregator: SequentialEmbeddingAggregatorProto,
        max_sequence_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding_aggregator = embedding_aggregator
        self.pe = torch.nn.Embedding(max_sequence_length, self.embedding_aggregator.embedding_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def reset_parameters(self) -> None:
        self.embedding_aggregator.reset_parameters()
        for _, param in self.pe.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(self, feature_tensors: TensorMap) -> torch.Tensor:
        seqs: torch.Tensor = self.embedding_aggregator(feature_tensors)
        assert seqs.dim() == 3
        batch_size, seq_len, embedding_dim = seqs.size()
        assert seq_len <= self.pe.embedding_dim, (
            "Sequence length is greater then positional embedding dim"
        )

        seqs *= embedding_dim**0.5
        seqs += self.pe.weight[:seq_len].unsqueeze(0).repeat(batch_size, 1, 1)
        seqs = self.dropout(seqs)
        return seqs
