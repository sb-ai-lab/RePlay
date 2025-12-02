from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TrainOutput:
    """
    A base class for storing outputs from models training stage

    :param loss: a tensor containing the calculated loss.\n
        It is important that the tensor contains a gradient to call back propagation from the outside.
    :param hidden_states: Tuple of `torch.Tensor`.\n
        One for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each layer.\n
        Expected shape: ``(batch_size, sequence_length, hidden_size)``.
    """

    loss: torch.Tensor
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None


@dataclass
class InferenceOutput:
    """
    A base class for storing outputs from models inference stage

    :param logits:
        Sequence of hidden-states at the output of the last layer of the model.\n
        Expected shape: ``(batch_size, sequence_length, hidden_size)``.
    :param hidden_states: Tuple of `torch.Tensor`
        (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each layer).\n
        Expected shape: ``(batch_size, sequence_length, hidden_size)``.
    """

    logits: torch.Tensor
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
