import contextlib
from typing import Protocol

import torch


class NormalizerProto(Protocol):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def reset_parameters(self) -> None: ...


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.
    Applies normalization across the last dimension and scales the output.
    """

    def __init__(self, embed_size: int, eps: float = 1e-5):
        """
        Args:
            embed_size (int): Dimension of the input features.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(embed_size))

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(self, inputs: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            seqs (Tensor): Input tensor of shape (batch, sequence_length, embed_size).

        Returns:
            Tensor: Normalized and scaled tensor.
        """
        norm = torch.sqrt(torch.mean(inputs**2, dim=-1, keepdim=True))
        return inputs / (norm + self.eps) * self.scale
