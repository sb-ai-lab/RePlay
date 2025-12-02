import contextlib
from typing import Protocol

import torch

from replay.data.nn.schema import TensorMap


class SequentialEmbeddingAggregatorProto(Protocol):
    """Class-protocol for working with embedding aggregation functions"""

    def forward(
        self,
        feature_tensors: TensorMap,
    ) -> torch.Tensor: ...

    @property
    def embedding_dim(self) -> int: ...

    def reset_parameters(self) -> None: ...


class SumAggregator(torch.nn.Module):
    """
    The class summarizes the incoming embeddings.
    Note that for successful aggregation, the dimensions of all embeddings must match.
    """

    def __init__(self, embedding_dim: int) -> None:
        """
        :param embedding_dim: The last dimension of incoming and outcoming embeddings.
        """
        super().__init__()
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        """The dimension of the output embedding"""
        return self._embedding_dim

    def reset_parameters(self) -> None:
        pass

    def forward(self, feature_tensors: TensorMap) -> torch.Tensor:
        """
        :param feature_tensors: a dictionary of tensors to sum up.
            The dimensions of all tensors in the dictionary must match.

        :returns: torch.Tensor. The last dimension of the tensor is ``embedding_dim``.
        """
        out = sum(feature_tensors.values())
        assert out.size(-1) == self.embedding_dim
        return out


class ConcatAggregator(torch.nn.Module):
    """
    The class concatenates incoming embeddings by the last dimension.

    If you need to concatenate several embeddings,
    then a linear layer will be applied to get the last dimension equal to ``embedding_dim``.\n
    If only one embedding comes to the input, then its last dimension is expected to be equal to ``embedding_dim``.
    """

    def __init__(
        self,
        input_embedding_dims: list[int],
        output_embedding_dim: int,
    ) -> None:
        """
        :param input_embedding_dims: Dimensions of incoming embeddings.
        :param output_embedding_dim: The dimension of the output embedding after concatenation.
        """
        super().__init__()
        self._embedding_dim = output_embedding_dim
        embedding_concat_size = sum(input_embedding_dims)
        self.feat_projection = None
        if len(input_embedding_dims) > 1:
            self.feat_projection = torch.nn.Linear(embedding_concat_size, self.embedding_dim)
        elif embedding_concat_size != self.embedding_dim:
            msg = f"Input embedding dim is not equal to embedding_dim ({embedding_concat_size} != {self.embedding_dim})"
            raise ValueError(msg)

    @property
    def embedding_dim(self) -> int:
        """The dimension of the output embedding"""
        return self._embedding_dim

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(self, feature_tensors: TensorMap) -> torch.Tensor:
        """
        To ensure the deterministic nature of the result,
        the embeddings are concatenated in the ascending order of the keys in the dictionary.

        :param feature_tensors: a dictionary of tensors to concatenate.

        :returns: The last dimension of the tensor is ``embedding_dim``.
        """
        # To maintain determinism, we concatenate the tensors in sorted order by names.
        sorted_names = sorted(feature_tensors.keys())
        out = torch.cat([feature_tensors[name] for name in sorted_names], dim=-1)
        if self.feat_projection is not None:
            out = self.feat_projection(out)
        assert out.size(-1) == self.embedding_dim
        return out
