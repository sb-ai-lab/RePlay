"""Model adapters: TensorMap models to flat positional-input graphs."""

from dataclasses import dataclass
from typing import Protocol

import torch

from replay.data.nn import TensorSchema

from ._manifest import (
    CANDIDATES_EXPLICIT,
    CANDIDATES_FULL_VOCAB,
    FeatureRecord,
)


@dataclass
class GraphInput:
    """One positional input of the exported graph."""

    name: str
    kind: str
    dtype: torch.dtype
    axes: list[str]


class InferenceAdapter(Protocol):
    """Builds a flat-input wrapper around a model's ``forward_inference``."""

    model_type: str

    def build(
        self,
        model: torch.nn.Module,
        schema: TensorSchema,
        candidates_policy: str,
    ) -> tuple[torch.nn.Module, list[GraphInput]]: ...


class _SasRecGraph(torch.nn.Module):
    def __init__(self, model, feature_names: list[str], candidates_as_input: bool):
        super().__init__()
        self._model = model
        self._feature_names = feature_names
        self._candidates_as_input = candidates_as_input

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        num_features = len(self._feature_names)
        feature_tensors = dict(zip(self._feature_names, inputs[:num_features], strict=True))
        padding_mask = inputs[num_features]
        candidates = inputs[-1] if self._candidates_as_input else None
        output = self._model.forward_inference(
            feature_tensors=feature_tensors,
            padding_mask=padding_mask,
            candidates_to_score=candidates,
        )
        return output["logits"]


class SasRecAdapter:
    model_type = "sasrec"

    def build(self, model, schema, candidates_policy):
        if candidates_policy not in (CANDIDATES_FULL_VOCAB, CANDIDATES_EXPLICIT):
            msg = f"Unknown candidates policy: {candidates_policy!r}"
            raise ValueError(msg)

        records = []
        for _, info in schema.items():
            if info.is_list:
                msg = (
                    f"Feature {info.name!r} is a categorical list; "
                    "list features are not exportable yet"
                )
                raise NotImplementedError(msg)
            records.append(FeatureRecord.from_tensor_info(info))
        feature_names = [record.name for record in records]

        graph_inputs = [
            GraphInput(
                name=name,
                kind="feature",
                dtype=torch.int64 if record.is_categorical else torch.float32,
                axes=["batch", "seq_len"],
            )
            for name, record in zip(feature_names, records, strict=True)
        ]
        graph_inputs.append(
            GraphInput(name="padding_mask", kind="padding_mask", dtype=torch.bool, axes=["batch", "seq_len"]),
        )
        if candidates_policy == CANDIDATES_EXPLICIT:
            graph_inputs.append(
                GraphInput(name="candidates", kind="candidates", dtype=torch.int64, axes=["candidates"]),
            )

        wrapper = _SasRecGraph(
            model=model,
            feature_names=feature_names,
            candidates_as_input=candidates_policy == CANDIDATES_EXPLICIT,
        )
        return wrapper, graph_inputs


class _TwoTowerQueryGraph(torch.nn.Module):
    def __init__(self, model, feature_names: list[str]):
        super().__init__()
        self._model = model
        self._feature_names = feature_names

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        num_features = len(self._feature_names)
        feature_tensors = dict(zip(self._feature_names, inputs[:num_features], strict=True))
        padding_mask = inputs[num_features]
        output = self._model.forward_inference(
            feature_tensors=feature_tensors,
            padding_mask=padding_mask,
            candidates_to_score=None,
        )
        return output["logits"]


class TwoTowerQueryAdapter:
    """Online query tower: full-catalog scoring only, per the two-graph layout."""

    model_type = "twotower_query"

    def build(self, model, schema, candidates_policy):
        if candidates_policy != CANDIDATES_FULL_VOCAB:
            msg = "The query-tower graph scores the external item matrix; candidates are not a graph input"
            raise ValueError(msg)

        records = [FeatureRecord.from_tensor_info(info) for _, info in schema.items()]
        feature_names = [record.name for record in records]
        graph_inputs = [
            GraphInput(
                name=name,
                kind="feature",
                dtype=torch.int64 if record.is_categorical else torch.float32,
                axes=["batch", "seq_len"],
            )
            for name, record in zip(feature_names, records, strict=True)
        ]
        graph_inputs.append(
            GraphInput(name="padding_mask", kind="padding_mask", dtype=torch.bool, axes=["batch", "seq_len"]),
        )
        wrapper = _TwoTowerQueryGraph(model=model, feature_names=feature_names)
        return wrapper, graph_inputs


_ADAPTERS: dict[str, InferenceAdapter] = {
    SasRecAdapter.model_type: SasRecAdapter(),
    TwoTowerQueryAdapter.model_type: TwoTowerQueryAdapter(),
}


def resolve_adapter(model) -> InferenceAdapter:
    from replay.nn.sequential.sasrec.model import SasRec
    from replay.nn.sequential.twotower.model import TwoTower

    if isinstance(model, SasRec):
        return _ADAPTERS[SasRecAdapter.model_type]
    if isinstance(model, TwoTower):
        return _ADAPTERS[TwoTowerQueryAdapter.model_type]
    msg = f"No compiled adapter for model type: {type(model).__name__}"
    raise TypeError(msg)
