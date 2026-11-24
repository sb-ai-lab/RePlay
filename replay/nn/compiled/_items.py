"""Item-embedding matrix artifact format for the two-artifact TwoTower topology.

This module is the on-disk contract and stays importable without torch:
serving only needs :class:`ItemEmbeddings` and :func:`load_item_embeddings`.
The torch-side producer (encoding a live model's catalog) lives in the
TwoTower compilation branch: ``replay.nn.compiled.export_item_embeddings``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ITEM_EMBEDDINGS_FILENAME = "item_embeddings.npy"
ITEM_EMBEDDINGS_LAYOUT_FILENAME = "item_embeddings.json"
ITEM_MATRIX_MODEL_TYPE = "twotower_query"


@dataclass
class ItemEmbeddings:
    """Full-catalog item-embedding matrix; row ``i`` is the embedding of item id ``i``."""

    matrix: np.ndarray
    feature_names: list[str]

    @property
    def num_items(self) -> int:
        return self.matrix.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.matrix.shape[1]

    def save(self, path: str | Path) -> None:
        """Persist the matrix (``.npy``) and its layout manifest into directory ``path``."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / ITEM_EMBEDDINGS_FILENAME, self.matrix)
        layout = {
            "model_type": ITEM_MATRIX_MODEL_TYPE,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
            "dtype": str(self.matrix.dtype),
            "feature_names": self.feature_names,
            "layout": "row i is the embedding of item id i",
        }
        (directory / ITEM_EMBEDDINGS_LAYOUT_FILENAME).write_text(json.dumps(layout, indent=2))


def load_item_embeddings(path: str | Path) -> ItemEmbeddings:
    """Load an :class:`ItemEmbeddings` artifact written by :meth:`ItemEmbeddings.save`."""
    directory = Path(path)
    layout_path = directory / ITEM_EMBEDDINGS_LAYOUT_FILENAME
    if not layout_path.is_file():
        msg = f"No item embeddings artifact in {directory!s}: expected {layout_path.name}"
        raise FileNotFoundError(msg)
    layout = json.loads(layout_path.read_text())
    matrix = np.load(directory / ITEM_EMBEDDINGS_FILENAME)
    if matrix.shape != (layout["num_items"], layout["embedding_dim"]):
        msg = f"Item embeddings shape {matrix.shape} disagrees with the layout manifest {layout}"
        raise ValueError(msg)
    return ItemEmbeddings(matrix=matrix, feature_names=layout["feature_names"])
