"""ONNX exporters behind a single seam.

The active exporter is a provisional choice; switching it must not
require changes outside this module.
"""

import io

import torch

LEGACY = "legacy"
DYNAMO = "dynamo"

ACTIVE_EXPORTER = LEGACY

_EXPORTERS = {}


def exporter(name):
    def decorator(fn):
        _EXPORTERS[name] = fn
        return fn

    return decorator


def export_onnx(
    module: torch.nn.Module,
    args: tuple,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset_version: int,
) -> bytes:
    """Export ``module(*args)`` to ONNX and return the serialized model."""
    fn = _EXPORTERS.get(ACTIVE_EXPORTER)
    if fn is None:
        msg = f"Unknown exporter: {ACTIVE_EXPORTER!r}"
        raise ValueError(msg)
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
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )
    return buffer.getvalue()


@exporter(DYNAMO)
def _export_dynamo(module, args, input_names, output_names, dynamic_axes, opset_version):
    buffer = io.BytesIO()
    torch.onnx.export(
        module,
        args,
        buffer,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=True,
    )
    return buffer.getvalue()
