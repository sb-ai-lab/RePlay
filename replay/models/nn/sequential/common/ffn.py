import contextlib
from typing import Literal, Protocol

import torch

from replay.data.nn import TensorMap
from replay.models.nn.utils import create_activation

from .normalization import RMSNorm


class PointWiseFeedForward(torch.nn.Module):
    """
    Point wise feed forward network layer

    Link: https://arxiv.org/pdf/1808.09781.pdf
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        activation: Literal["relu", "gelu"] = "gelu",
    ) -> None:
        """
        :param hidden_size: Hidden size.
        :param dropout: Dropout rate.
        """
        super().__init__()

        self.conv1 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.activation = create_activation(activation)
        self.conv2 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout)

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(self, input_embeddings: torch.LongTensor) -> torch.LongTensor:
        """
        :param inputs: Query feature vector.

        :returns: Output tensors.
        """
        x: torch.Tensor = self.conv1(input_embeddings.transpose(-1, -2))
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = x.transpose(-1, -2)
        x += input_embeddings

        return x


class SwiGLU(torch.nn.Module):
    """
    SwiGLU Activation Function.
    Combines the Swish activation with Gated Linear Units.
    """

    def __init__(self, embedding_size: int, multiplier: float = 2.0):
        """
        Args:
            embedding_size (int): Dimension of the input features.
            multiplier (float): multipler for embedding_size to get dimensionality of hidden layer
        """
        super().__init__()
        hidden_size = int(embedding_size * multiplier)
        if hidden_size == 0:
            msg = f"hidden_dim should not be equal to `0`. Got {multiplier=} and {embedding_size=}"
            raise ValueError(msg)

        # Intermediate projection layers
        # Typically, SwiGLU splits the computation into two parts
        self.WG = torch.nn.Linear(embedding_size, hidden_size)
        self.W1 = torch.nn.Linear(embedding_size, hidden_size)
        self.W2 = torch.nn.Linear(hidden_size, embedding_size)

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(
        self,
        feature_tensors: TensorMap,  # noqa: ARG002
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for SwiGLU.

        Args:
            seqs (Tensor): Input tensor of shape (batch, sequence_length, embed_size).

        Returns:
            Tensor: Output tensor after applying SwiGLU.
        """
        # Apply the gates
        activation = torch.nn.functional.silu(self.WG(input_embeddings))  # Activation part
        linear = self.W1(input_embeddings)  # Linear part
        return self.W2(activation * linear)  # Element-wise multiplication and projection


class SwiGLUEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_size: int,
    ) -> None:
        super().__init__()
        self.resnet_block1 = SwiGLU(embed_size)
        self.layernorm1 = RMSNorm(embed_size)
        self.resnet_block2 = SwiGLU(embed_size)
        self.layernorm2 = RMSNorm(embed_size)

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(
        self,
        feature_tensors: TensorMap,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        x = self.layernorm1(self.resnet_block1(feature_tensors, input_embeddings) + input_embeddings)
        x = self.layernorm2(self.resnet_block2(feature_tensors, x) + x)
        return x


class SparseMultiHeadEncoderProto(Protocol):
    def forward(
        self,
        feature_tensors: TensorMap,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor: ...


class SparseMultiHead(torch.nn.Module):
    """
    Sparse multi head torch module:

    One head (expert) per each `gate_feature_name` feature value.
    Based on the `gate_feature_name` feature value - route to the corresponding expert.
    """

    def __init__(self, encoder: SparseMultiHeadEncoderProto, num_heads: int):
        """
        Args:
            :param encoder: encoder object.
            :param n_heads: number of heads (experts).
        """
        super().__init__()
        self.ff = torch.nn.ModuleList([encoder for _ in range(num_heads)])

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(
        self,
        feature_tensors: TensorMap,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        ff_out = [
            module(
                feature_tensors=feature_tensors,
                input_embeddings=input_embeddings,
            ).unsqueeze(-2)
            for module in self.ff
        ]
        ff_out.append(torch.zeros_like(ff_out[0]))
        ff_out = torch.concat(ff_out, dim=-2)

        batch_size, sequence_length = ff_out.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(-1, sequence_length, -1)
        sequence_indices = torch.arange(sequence_length).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1)

        ff_out = ff_out[
            batch_indices.flatten(),
            sequence_indices.flatten(),
        ].view(batch_size, sequence_length, -1)
        return ff_out


class SparseSwiGLU(SparseMultiHead):
    """
    SparseMultiHead with SwiGLU as encoder block.
    """

    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        multiplier: float = 2.0,
    ):
        """
        Args:
            :param num_heads: number of heads (experts).
            :param embedding_dim: Dimension of the input features.
            :param multiplier: multipler for embedding_dim to get dimensionality of hidden layer.
        """
        super().__init__(
            SwiGLU(embedding_dim, multiplier),
            num_heads,
        )


class SparsePointWiseFeedForward(SparseMultiHead):
    """
    SparseMultiHead with PointWiseFeedForward as encoder block.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        dropout: float,
        activation: Literal["relu", "gelu"] = "gelu",
    ) -> None:
        """
        Args:
            :param n_heads: number of heads (experts).
            :param hidden_size: Dimension of the input features.
            :param dropout: Dropout rate.
            :param activation: Non-linear activation function.
        """
        super().__init__(
            PointWiseFeedForward(
                hidden_size=hidden_size,
                dropout=dropout,
                activation=activation,
            ),
            num_heads,
        )
