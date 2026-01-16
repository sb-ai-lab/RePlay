import warnings
from typing import Callable, Literal, Tuple

import torch


def warning_is_not_none(msg: str) -> Callable:
    def checker(value: Tuple[torch.Tensor, str]) -> bool:
        if value[0] is not None:
            warnings.warn(msg.format(value[1]), RuntimeWarning, stacklevel=2)
            return False
        return True

    return checker


def create_activation(
    activation: Literal["relu", "gelu", "sigmoid"],
) -> torch.nn.Module:
    """The function of creating an activation function based on its name"""
    if activation == "relu":
        return torch.nn.ReLU()
    if activation == "gelu":
        return torch.nn.GELU()
    if activation == "sigmoid":
        return torch.nn.Sigmoid()
    msg = "Expected to get activation relu/gelu/sigmoid"
    raise ValueError(msg)
