import warnings
from typing import Iterable, Self, cast

import torch

from replay.data.nn import TensorMap

from .base import LogitsCallback, LossInfo, LossOutput, LossProto

Weights = dict[str, torch.Tensor | float]
Losses = Iterable[torch.nn.Module] | dict[str, torch.nn.Module] | torch.nn.ModuleDict


class ComposedLoss(torch.nn.Module):
    def __init__(
        self: Self,
        losses: Losses,
        weights: Weights | None = None,
        loss_name: str = "ComposedLoss",
    ) -> None:
        super().__init__()

        self.losses: torch.nn.ModuleDict = self._handle_losses(losses)
        self.weights: Weights = self._handle_weights(weights)
        self._logits_callback: LogitsCallback | None = None
        self.loss_name: str = loss_name

    def _handle_losses(self: Self, losses: Losses) -> torch.nn.ModuleDict:
        if not isinstance(losses, torch.nn.ModuleDict):
            if isinstance(losses, dict):
                for loss in cast(dict, losses.values()):
                    if not isinstance(loss, torch.nn.Module):
                        msg: str = f"Unsupported type of loss. Must be `Module`. Got: {type(loss)=}."
                        raise TypeError(msg)
                losses_dict: dict[str, torch.nn.Module] = cast(dict[str, torch.nn.Module], losses)
            else:
                losses_dict: dict[str, torch.nn.Module] = {}
                for loss in iter(losses):
                    casted: LossProto = cast(LossProto, loss)
                    name: str = casted.loss_name
                    if name in losses_dict:
                        msg: str = f"Loss names must be unique. Got {name} twice."
                        raise KeyError(name)
                    losses_dict[name] = loss
            losses = torch.nn.ModuleDict(losses_dict)

        if not isinstance(losses, torch.nn.ModuleDict):
            msg: str = f"Unsupported type of `losses`. Must be `dict` or `ModuleDict`. Got {type(losses)=}."
            raise TypeError(msg)

        if len(losses) < 1:
            msg: str = "Empty losses are not supported."
            raise ValueError(msg)

        return cast(torch.nn.ModuleDict, losses)

    def _handle_weights(self: Self, weights: Weights | None) -> Weights:
        if weights is None:
            weights = {}
        elif not isinstance(weights, dict):
            msg: str = f"Unsupported type of `weights`. Must be `dict`. Got: {type(weights)=}."
            raise TypeError(msg)

        for name, weight in cast(dict, weights):
            if name not in self.losses:
                msg: str = f"Unknown name of weight: {name}."
                warnings.warn(msg, stacklevel=2)
                continue
            if isinstance(weight, float):
                continue
            elif isinstance(weight, torch.Tensor):
                assert torch.is_tensor(weight)
                if torch.numel(weight) > 1:
                    msg: str = f"Too many values in weight: {torch.numel(weight)=}."
                    raise ValueError(msg)
                continue
            else:
                msg: str = f"Unsupported type of weight value. Must be `float` or `Tensor`. Got: {type(weight)=}."
                raise TypeError(msg)

        return cast(Weights, weights)

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

    def _compute_raw_losses(
        self: Self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> dict[str, torch.Tensor]:
        raw_losses: dict[str, torch.Tensor] = {}
        for name, loss in self.losses.items():
            value, _ = loss(
                model_embeddings,
                feature_tensors,
                positive_labels,
                negative_labels,
                padding_mask,
                target_padding_mask,
            )
            raw_losses[name] = value
        return raw_losses

    def _apply_weights(self: Self, raw_losses: dict[str, torch.Tensor]) -> torch.Tensor:
        losses_list: list[torch.Tensor] = []

        for name, value in raw_losses.items():
            weight = self.weights.get(name, 1.0)
            losses_list.append(weight * value[None])

        losses: torch.Tensor = torch.cat(losses_list)
        return torch.sum(losses)

    def _detach_dict(self: Self, raw_losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: value.detach() for name, value in raw_losses.items()}

    def forward(
        self: Self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
        return_info: bool = False,
    ) -> LossOutput:
        raw_losses: dict[str, torch.Tensor] = self._compute_raw_losses(
            model_embeddings=model_embeddings,
            feature_tensors=feature_tensors,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            padding_mask=padding_mask,
            target_padding_mask=target_padding_mask,
        )

        loss: torch.Tensor = self._apply_weights(raw_losses)

        if return_info:
            base_info: dict[str, torch.Tensor] = self._detach_dict(raw_losses)
            info: LossInfo = {self.loss_name: loss.detach(), **base_info}
            return (loss, info)
        else:
            return (loss, None)
