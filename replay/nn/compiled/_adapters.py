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


def feature_axes(record: FeatureRecord) -> list[str]:
    """Axis names of one feature input; ``tensor_dim``/list width beyond 1 adds a static axis."""
    if record.is_list:
        return ["batch", "seq_len", "list_len"]
    if record.is_categorical or record.tensor_dim in (None, 1):
        return ["batch", "seq_len"]
    return ["batch", "seq_len", "tensor_dim"]


class InferenceAdapter(Protocol):
    """Builds a flat-input wrapper around a model's ``forward_inference``."""

    model_type: str
    output_name: str

    def build(
        self,
        model: torch.nn.Module,
        schema: TensorSchema,
        candidates_policy: str,
    ) -> tuple[torch.nn.Module, list[GraphInput], list[FeatureRecord]]: ...


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
    output_name = "logits"

    def build(self, model, schema, candidates_policy):
        if candidates_policy not in (CANDIDATES_FULL_VOCAB, CANDIDATES_EXPLICIT):
            msg = f"Unknown candidates policy: {candidates_policy!r}"
            raise ValueError(msg)

        records = [FeatureRecord.from_tensor_info(info) for info in schema.values()]
        feature_names = [record.name for record in records]

        graph_inputs = [
            GraphInput(
                name=name,
                kind="feature",
                dtype=torch.int64 if record.is_categorical else torch.float32,
                axes=feature_axes(record),
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
        return wrapper, graph_inputs, records


class _TwoTowerQueryGraph(torch.nn.Module):
    """Query tower only, per the two-artifact topology: the graph outputs
    last-position query embeddings; the item-embedding matrix stays an
    external artifact and the scorer lives in the serving layer."""

    def __init__(self, model, feature_names: list[str]):
        super().__init__()
        self._model = model
        self._feature_names = feature_names

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        num_features = len(self._feature_names)
        feature_tensors = dict(zip(self._feature_names, inputs[:num_features], strict=True))
        padding_mask = inputs[num_features]
        hidden_states = self._model.body.query_tower(feature_tensors, padding_mask)
        if self._model.context_merger is not None:
            hidden_states = self._model.context_merger(
                model_hidden_state=hidden_states,
                feature_tensors=feature_tensors,
            )
        return hidden_states[:, -1, :].contiguous()


class TwoTowerQueryAdapter:
    """Online query tower of the two-artifact layout (item matrix is exported separately)."""

    model_type = "twotower_query"
    output_name = "query_embeddings"

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
                axes=feature_axes(record),
            )
            for name, record in zip(feature_names, records, strict=True)
        ]
        graph_inputs.append(
            GraphInput(name="padding_mask", kind="padding_mask", dtype=torch.bool, axes=["batch", "seq_len"]),
        )
        wrapper = _TwoTowerQueryGraph(model=model, feature_names=feature_names)
        return wrapper, graph_inputs, records


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
