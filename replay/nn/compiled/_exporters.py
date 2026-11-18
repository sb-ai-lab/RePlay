"""ONNX exporters behind a single seam.

The active exporter is a provisional choice; switching it must not
require changes outside this module. Axis specs arrive in a neutral
form (name -> {dim index: axis name}, inputs and outputs alike) and
each exporter translates them to its native kwargs.
"""

import io
from contextlib import contextmanager

import torch

LEGACY = "legacy"
DYNAMO = "dynamo"
AUTO = "auto"

ACTIVE_EXPORTER = LEGACY

_EXPORTERS = {}


def exporter(name):
    def decorator(fn):
        _EXPORTERS[name] = fn
        return fn

    return decorator


def select_exporter(module: torch.nn.Module, preference: str = AUTO) -> str:
    """Resolve the export route for ``module``.

    ``legacy`` is the fast path (10-20x faster export), but ``aten::rms_norm``
    is unsupported at opset 20, so any graph containing ``torch.nn.RMSNorm``
    must go through ``dynamo``.
    """
    if preference not in (AUTO, LEGACY, DYNAMO):
        msg = f"Unknown exporter: {preference!r}"
        raise ValueError(msg)
    if preference != AUTO:
        return preference
    has_rms_norm = any(isinstance(submodule, torch.nn.RMSNorm) for submodule in module.modules())
    return DYNAMO if has_rms_norm else LEGACY


@contextmanager
def _scoped_mha_fastpath(enabled: bool):
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(enabled)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def export_onnx(
    module: torch.nn.Module,
    args: tuple,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset_version: int,
    exporter: str | None = None,
) -> bytes:
    """Export ``module(*args)`` to ONNX and return the serialized model."""
    if exporter is None:
        exporter = ACTIVE_EXPORTER
    fn = _EXPORTERS.get(exporter)
    if fn is None:
        msg = f"Unknown exporter: {exporter!r}"
        raise ValueError(msg)
    with _scoped_mha_fastpath(False), torch.no_grad():
        return fn(module, args, input_names, output_names, dynamic_axes, opset_version)


@exporter(LEGACY)
def _export_legacy(module, args, input_names, output_names, dynamic_axes, opset_version):
    buffer = io.BytesIO()
    torch.onnx.export(
        module,
        args,
        buffer,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes or None,
        opset_version=opset_version,
        dynamo=False,
    )
    return buffer.getvalue()


@exporter(DYNAMO)
def _export_dynamo(module, args, input_names, output_names, dynamic_axes, opset_version):
    buffer = io.BytesIO()
    # dynamo rejects `dynamic_axes`; per-tensor shape specs for a varargs
    # module travel as a list holding one tuple with one dict per tensor
    dynamic_shapes = [tuple(dynamic_axes.get(name) or None for name in input_names)]
    torch.onnx.export(
        module,
        args,
        buffer,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=opset_version,
        dynamo=True,
    )
    return buffer.getvalue()
