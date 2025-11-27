from typing import Protocol

import torch


class NormalizerProto(Protocol):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def reset_parameters(self) -> None: ...
