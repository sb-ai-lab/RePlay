"""Shared vocabulary of the per-model compilation branches."""

from dataclasses import dataclass
from typing import Protocol

import torch

from replay.data.nn import TensorSchema
from replay.nn.compiled._manifest import FeatureRecord


@dataclass
class GraphInput:
    """One positional input of the exported graph."""

    name: str
    kind: str
    dtype: torch.dtype
    axes: list[str]


def feature_axes(record: FeatureRecord) -> list[str]:
    """Axis names of one feature input; ``tensor_dim``/list width beyond 1 adds a static axis."""
    if record.is_list:
        return ["batch", "seq_len", "list_len"]
    if record.is_categorical or record.tensor_dim in (None, 1):
        return ["batch", "seq_len"]
    return ["batch", "seq_len", "tensor_dim"]


class InferenceAdapter(Protocol):
    """Everything the generic pipeline needs from one model family.

    A branch is the single place where a model family's compilation
    specifics live: the flat-input wrapper around ``forward_inference``,
    the candidates policies it supports, and how its checkpoints rebuild.
    """

    model_type: str
    output_name: str
    model_cls: type
    uses_item_features_reader: bool

    def build(
        self,
        model: torch.nn.Module,
        schema: TensorSchema,
        candidates_policy: str,
    ) -> tuple[torch.nn.Module, list[GraphInput], list[FeatureRecord]]: ...
