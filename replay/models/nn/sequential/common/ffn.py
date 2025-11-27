import contextlib
from typing import Literal

import torch

from replay.data.nn import TensorMap
from replay.models.nn.utils import create_activation


class PointWiseFeedForward(torch.nn.Module):
    """
    Point wise feed forward network layer

    Link: https://arxiv.org/pdf/1808.09781.pdf
    """

    def __init__(
        self,
        embedding_dim: int,
        dropout: float,
        activation: Literal["relu", "gelu"] = "gelu",
    ) -> None:
        """
        :param embedding_dim: Dimension of the input features.
        :param dropout: probability of an element to be zeroed.
        :param activation: the name of the activation function.
            Possible values are ``"relu"``, ``"gelu"``.
        """
        super().__init__()

        self.conv1 = torch.nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.activation = create_activation(activation)
        self.conv2 = torch.nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
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

    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        :param embedding_dim: Dimension of the input features.
        :param hidden_dim: Dimension of hidden layer.
        """
        super().__init__()
        # Intermediate projection layers
        # Typically, SwiGLU splits the computation into two parts
        self.WG = torch.nn.Linear(embedding_dim, hidden_dim)
        self.W1 = torch.nn.Linear(embedding_dim, hidden_dim)
        self.W2 = torch.nn.Linear(hidden_dim, embedding_dim)

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(
        self,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for SwiGLU.

        :param input_embeddings: Input tensor of shape (batch_size, sequence_length, embedding_dim).

        :returns: Tensor. Output tensor of shape (batch_size, sequence_length, embedding_dim) after applying SwiGLU.
        """
        # Apply the gates
        activation = torch.nn.functional.silu(self.WG(input_embeddings))  # Activation part
        linear = self.W1(input_embeddings)  # Linear part
        return self.W2(activation * linear)  # Element-wise multiplication and projection


class SwiGLUEncoder(torch.nn.Module):
    """
    MLP block consists of SwiGLU Feed-Forward network followed by a RMSNorm layer with skip connection.

    RMSNorm paper: https://arxiv.org/pdf/1910.07467.
    """

    def __init__(self, embedding_dim: int) -> None:
        """
        :param embedding_dim: Dimension of the input features.
        """
        super().__init__()
        self.sw1 = SwiGLU(embedding_dim)
        self.norm1 = torch.nn.RMSNorm(embedding_dim)
        self.sw2 = SwiGLU(embedding_dim)
        self.norm2 = torch.nn.RMSNorm(embedding_dim)

    def reset_parameters(self) -> None:
        self.sw1.reset_parameters()
        self.sw2.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def forward(
        self,
        feature_tensors: TensorMap,  # noqa: ARG002
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param input_embeddings: Input tensor of shape (batch_size, sequence_length, embedding_dim).
        :returns: torch.Tensor: Output tensor after processing through the MLP.
        """
        x = self.norm1(self.sw1(input_embeddings) + input_embeddings)
        x = self.norm2(self.sw2(x) + x)
        return x
