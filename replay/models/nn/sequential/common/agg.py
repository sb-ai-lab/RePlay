import contextlib
from typing import Protocol

import torch

from replay.data.nn.schema import TensorMap


class SequentialEmbeddingAggregatorProto(Protocol):
    def forward(
        self,
        feature_tensors: TensorMap,
    ) -> torch.Tensor: ...

    def reset_parameters(self) -> None: ...


class SumAggregator(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim

    def reset_parameters(self) -> None:
        return

    def forward(self, feature_tensors: TensorMap) -> torch.Tensor:
        out = sum(feature_tensors.values())
        assert out.size(-1) == self._embedding_dim
        return out


class ConcatAggregator(torch.nn.Module):
    def __init__(
        self,
        embed_sizes: list[int],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        assert embed_sizes
        self._embedding_dim = embedding_dim
        embedding_concat_size = sum(embed_sizes)
        self.feat_projection = None
        if len(embed_sizes) > 1:
            self.feat_projection = torch.nn.Linear(embedding_concat_size, embedding_dim)
        elif embedding_concat_size != embedding_dim:
            msg = f"Input embedding dim is not equal to embedding_dim ({embedding_concat_size} != {embedding_dim})"
            raise ValueError(msg)

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(self, feature_tensors: TensorMap) -> torch.Tensor:
        # To maintain determinism, we concatenate the tensors in sorted order by names.
        sorted_names = sorted(feature_tensors.keys())
        out = torch.cat([feature_tensors[name] for name in sorted_names], dim=-1)
        if self.feat_projection is not None:
            out = self.feat_projection(out)
        assert out.size(-1) == self._embedding_dim
        return out
