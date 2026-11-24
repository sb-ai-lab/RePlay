"""TwoTower compilation branch: online query-tower graph + offline item matrix.

The two-artifact topology: the online graph encodes users into query
embeddings, while the item side is exported separately as a full-catalog
embedding matrix (:func:`export_item_embeddings`); scoring
(``query_embeddings @ matrix.T``) and candidate handling stay in the
serving layer.
"""

import numpy as np
import torch

from replay.nn.compiled._items import ItemEmbeddings
from replay.nn.compiled._manifest import (
    CANDIDATES_FULL_VOCAB,
    FeatureRecord,
)
from replay.nn.sequential.twotower.model import TwoTower

from ._base import GraphInput, feature_axes


class _TwoTowerQueryGraph(torch.nn.Module):
    """Query tower only: the graph outputs last-position query embeddings;
    the item-embedding matrix stays an external artifact and the scorer
    lives in the serving layer."""

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
    """Compiles the online query tower of the two-artifact layout.

    Candidates policy: ``full_vocab`` only, by construction rather than by
    arbitrary restriction. The graph emits query embeddings ``[B, E]`` and
    never scores items itself — scoring against the external item matrix is
    a serving-side matmul, and restricting it to a candidate subset is a row
    gather over that matrix (``matrix[candidates]``). Candidate selection
    therefore needs no graph input and no recompile; the ``explicit``
    policy exists for graphs whose own output axis is the scored set.
    """

    model_type = "twotower_query"
    output_name = "query_embeddings"
    model_cls = TwoTower
    uses_item_features_reader = True

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


def export_item_embeddings(model: torch.nn.Module) -> ItemEmbeddings:
    """Encode the full item catalog with a ``TwoTower`` model's item tower.

    The matrix is the offline artifact of the two-artifact topology: the online
    query-tower graph is exported by :func:`replay.nn.compiled.compile`, scoring
    (``query_embeddings @ matrix.T``) and ANN search stay in the serving layer.

    :param model: a ``TwoTower`` module (eval mode is enforced).
    :return: an :class:`ItemEmbeddings` with a ``[num_items, embedding_dim]`` float matrix.
    """
    if not isinstance(model, TwoTower):
        msg = f"Item embeddings are exported from TwoTower models, got: {type(model).__name__}"
        raise TypeError(msg)
    model.eval()
    with torch.no_grad():
        matrix = model.body.item_tower(None)
    return ItemEmbeddings(
        matrix=matrix.detach().numpy().astype(np.float32, copy=False),
        feature_names=list(model.body.item_tower.feature_names),
    )
