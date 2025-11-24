from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TrainOutput:
    loss: torch.Tensor
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None


@dataclass
class InferenceOutput:
    logits: torch.Tensor
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
