"""Offline item-embedding matrix artifact for the two-artifact TwoTower topology (D3)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ITEM_EMBEDDINGS_FILENAME = "item_embeddings.npy"
ITEM_EMBEDDINGS_LAYOUT_FILENAME = "item_embeddings.json"


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
            "model_type": "twotower_query",
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
            "dtype": str(self.matrix.dtype),
            "feature_names": self.feature_names,
            "layout": "row i is the embedding of item id i",
        }
        (directory / ITEM_EMBEDDINGS_LAYOUT_FILENAME).write_text(json.dumps(layout, indent=2))


def export_item_embeddings(model: torch.nn.Module) -> ItemEmbeddings:
    """Encode the full item catalog with a ``TwoTower`` model's item tower.

    The matrix is the offline artifact of the two-artifact topology: the online
    query-tower graph is exported by :func:`replay.nn.compiled.compile`, scoring
    (``query_embeddings @ matrix.T``) and ANN search stay in the serving layer.

    :param model: a ``TwoTower`` module (eval mode is enforced).
    :return: an :class:`ItemEmbeddings` with a ``[num_items, embedding_dim]`` float matrix.
    """
    from replay.nn.sequential.twotower.model import TwoTower

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
