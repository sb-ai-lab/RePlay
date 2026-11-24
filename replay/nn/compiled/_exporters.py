"""ONNX export through the dynamo route.

Axis specs arrive in a neutral form (name -> {dim index: axis name},
inputs and outputs alike) and are translated to dynamo's native kwargs.
"""

import io
from contextlib import contextmanager

import torch

DYNAMO = "dynamo"


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
) -> bytes:
    """Export ``module(*args)`` to ONNX and return the serialized model."""
    buffer = io.BytesIO()
    # dynamo rejects `dynamic_axes`; per-tensor shape specs for a varargs
    # module travel as a list holding one tuple with one dict per tensor
    dynamic_shapes = [tuple(dynamic_axes.get(name) or None for name in input_names)]
    with _scoped_mha_fastpath(False), torch.no_grad():
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
