import warnings
from typing import Optional, Self, cast

import torch

from replay.data.nn import TensorMap

from .base import LogitsCallback, LossProto

Weights = dict[str, torch.Tensor | float]


class ComposedLoss(torch.nn.Module):
    def __init__(
        self: Self, losses: dict[str, torch.nn.Module] | torch.nn.ModuleDict, weights: Weights | None = None
    ) -> None:
        super().__init__()

        if isinstance(losses, dict):
            for loss in cast(dict, losses.values()):
                if not isinstance(loss, torch.nn.Module):
                    msg: str = f"Unsupported type of loss. Must be `Module`. Got: {type(loss)=}."
                    raise TypeError(msg)
            losses = torch.nn.ModuleDict(losses)

        if not isinstance(losses, torch.nn.ModuleDict):
            msg: str = f"Unsupported type of `losses`. Must be `dict` or `ModuleDict`. Got {type(losses)=}."
            raise TypeError(msg)

        if len(losses) < 1:
            msg: str = "Empty losses are not supported."
            raise ValueError(msg)

        self.losses: torch.nn.ModuleDict = cast(torch.nn.ModuleDict, losses)

        if weights is None:
            weights = {}

        if not isinstance(weights, dict):
            msg: str = f"Unsupported type of `weights`. Must be `dict`. Got: {type(weights)=}."

        for name, weight in cast(dict, weights):
            if name not in self.losses:
                msg: str = f"Unknown name of weight: {name}."
                warnings.warn(msg, stacklevel=2)
                continue
            if isinstance(weight, float):
                continue
            elif isinstance(weight, torch.Tensor):
                assert torch.is_tensor(weight)
                continue
            else:
                msg: str = f"Unsupported type of weight value. Must be `float` or `Tensor`. Got: {type(weight)=}."
                raise TypeError(msg)

        self.weights: dict[str, torch.Tensor | float] = cast(Weights, weights)

        self._logits_callback: Optional[LogitsCallback] = None

    @property
    def logits_callback(self: Self) -> LogitsCallback:
        if self._logits_callback is None:
            msg: str = "No `logits_callback` provided"
            raise NotImplementedError(msg)
        return self._logits_callback

    @logits_callback.setter
    def logits_callback(self: Self, func: LogitsCallback) -> None:
        self._logits_callback = func

        for loss in self.losses.values():
            casted = cast(LossProto, loss)
            casted.logits_callback = func

    def forward(
        self: Self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        losses = 0.0
        for name, loss in self.losses.items():
            loss_weight = self.weights.get(name, 1.0)
            loss_value: torch.Tensor = loss(
                model_embeddings,
                feature_tensors,
                positive_labels,
                negative_labels,
                padding_mask,
                target_padding_mask,
            )
            losses = losses + loss_weight * loss_value
        return cast(torch.Tensor, losses)
