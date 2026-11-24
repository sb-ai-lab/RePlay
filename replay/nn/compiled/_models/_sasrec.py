"""SasRec compilation branch: ``forward_inference`` logits graph."""

import torch

from replay.nn.compiled._manifest import (
    CANDIDATES_EXPLICIT,
    CANDIDATES_FULL_VOCAB,
    FeatureRecord,
)
from replay.nn.sequential.sasrec.model import SasRec

from ._base import GraphInput, feature_axes


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
    """Compiles ``SasRec`` into a sequential logits graph.

    Candidates policies (how the scored item set is chosen at inference):

    - ``full_vocab`` — the graph scores every item id: the output logits
      span the whole vocabulary, and the item table is baked into the graph
      as constant initializers. Best for full-catalog ranking.
    - ``explicit`` — an int64 ``candidates`` tensor becomes an extra graph
      input: the graph scores exactly the passed ids, and the output's last
      axis is the ``num_candidates`` fixed at compile time. Best when a
      retriever pre-selects a small candidate set.

    Both policies change only the graph boundary; the transformer body and
    the parity contract are identical.
    """

    model_type = "sasrec"
    output_name = "logits"
    model_cls = SasRec
    uses_item_features_reader = False

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
