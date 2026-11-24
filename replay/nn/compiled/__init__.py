"""ONNX and OpenVINO compilation of ``replay.nn`` inference graphs.

Layout: the generic pipeline lives in ``_core`` (compile/load), with
``_exporters``, ``_manifest``, ``_swap``, ``_encoding``, ``_checkpoint``
and the artifact formats in ``_items``; everything model-specific lives
under ``_models`` — one module per model family, dispatched by registry.

Symbol imports are lazy so the serving path — ``load``, ``encode_inputs``,
``ItemEmbeddings`` and friends — works on machines without torch;
only ``compile`` and the config-capture helpers require the training stack.
"""

from importlib import import_module

_EXPORTS = {
    "CompiledModel": "._core",
    "ItemEmbeddings": "._items",
    "ModelConfig": "._checkpoint",
    "capture_config": "._checkpoint",
    "compile": "._core",
    "encode_inputs": "._encoding",
    "export_item_embeddings": "._models._twotower",
    "load": "._core",
    "load_item_embeddings": "._items",
    "read_config": "._checkpoint",
    "restore_from_checkpoint": "._checkpoint",
    "save_config": "._checkpoint",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name], __name__), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    return __all__
