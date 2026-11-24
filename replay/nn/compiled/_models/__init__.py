"""Per-model compilation branches and the registry that dispatches to them.

Adding compiled support for a model family means one module here plus one
registry row: the generic pipeline (``_core``), the exporter seam
(``_exporters``), the manifest, the graph surgery (``_swap``), the encoding
helper (``_encoding``) and the config capture (``_checkpoint``) all stay
model-agnostic and talk to branches only through
:class:`_base.InferenceAdapter`.

This package requires torch; the torch-free serving path never imports it.
"""

from ._base import GraphInput, InferenceAdapter, feature_axes
from ._sasrec import SasRecAdapter
from ._twotower import TwoTowerQueryAdapter, export_item_embeddings

_BRANCHES = (SasRecAdapter(), TwoTowerQueryAdapter())
_BY_TYPE = {branch.model_type: branch for branch in _BRANCHES}


def resolve_branch(model_or_cls) -> InferenceAdapter:
    """The compilation branch matching a model instance or class (isinstance dispatch)."""
    cls = model_or_cls if isinstance(model_or_cls, type) else model_or_cls.__class__
    for branch in _BRANCHES:
        if issubclass(cls, branch.model_cls):
            return branch
    msg = f"No compiled adapter for model type: {cls.__name__}"
    raise TypeError(msg)


def branch_by_type(model_type: str) -> InferenceAdapter:
    """The compilation branch registered under a manifest ``model_type``."""
    try:
        return _BY_TYPE[model_type]
    except KeyError:
        msg = f"No rebuild support for model type: {model_type!r}"
        raise ValueError(msg) from None
