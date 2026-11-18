"""Generic compilation core: :func:`compile`, :class:`CompiledModel` and :func:`load`."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from replay.utils import OPENVINO_AVAILABLE

from ._adapters import resolve_adapter
from ._exporters import AUTO, export_onnx, select_exporter
from ._manifest import (
    CANDIDATES_EXPLICIT,
    CANDIDATES_FULL_VOCAB,
    MANIFEST_FILENAME,
    ExportManifest,
)

MODE_DYNAMIC_BATCH = "dynamic_batch_size"
MODE_BATCH = "batch"
_MODES = (MODE_DYNAMIC_BATCH, MODE_BATCH)

DEFAULT_OPSET_VERSION = 20
ONNX_FILENAME = "model.onnx"
AXIS_BATCH = "batch"
AXIS_SEQUENCE = "seq_len"
_DYNAMIC_AXES = frozenset({AXIS_BATCH, AXIS_SEQUENCE, "candidates"})


def _check_conditional_stack() -> None:
    if not OPENVINO_AVAILABLE:
        msg = (
            "replay.nn.compiled requires torch, onnx and openvino; "
            "install the missing packages manually (see the README install notes)"
        )
        raise ImportError(msg)


def _stack_versions() -> dict[str, str]:
    import onnx
    import openvino

    return {
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "openvino": openvino.__version__,
    }


def _build_sample(schema, graph_inputs, batch_size, max_seq_len, num_candidates):
    sample = {}
    for graph_input in graph_inputs:
        if graph_input.kind == "feature":
            info = schema[graph_input.name]
            if info.is_cat:
                tensor = torch.randint(0, info.cardinality, (batch_size, max_seq_len))
            elif info.tensor_dim == 1:
                tensor = torch.randn(batch_size, max_seq_len)
            else:
                tensor = torch.randn(batch_size, max_seq_len, info.tensor_dim)
        elif graph_input.kind == "padding_mask":
            tensor = torch.ones(batch_size, max_seq_len, dtype=torch.bool)
        else:
            tensor = torch.arange(num_candidates)
        sample[graph_input.name] = tensor.to(graph_input.dtype)
    return sample


def _dynamic_axes(graph_inputs, mode, output_name):
    axes = {}
    for graph_input in graph_inputs:
        dims = {
            index: axis
            for index, axis in enumerate(graph_input.axes)
            if axis in _DYNAMIC_AXES and not (axis == AXIS_BATCH and mode == MODE_BATCH)
        }
        if dims:
            axes[graph_input.name] = dims
    if mode == MODE_DYNAMIC_BATCH:
        axes[output_name] = {0: AXIS_BATCH}
    return axes


def _input_shapes(graph_inputs, sample, mode):
    shapes = {}
    for graph_input in graph_inputs:
        tensor = sample[graph_input.name]
        shapes[graph_input.name] = [
            axis if axis in _DYNAMIC_AXES and not (axis == AXIS_BATCH and mode == MODE_BATCH) else tensor.shape[index]
            for index, axis in enumerate(graph_input.axes)
        ]
    return shapes


def _compile_openvino(onnx_bytes, manifest, num_threads):
    import openvino as ov

    core = ov.Core()
    if num_threads is not None:
        core.set_property("CPU", {"INFERENCE_NUM_THREADS": num_threads})
    # convert_model accepts neither ModelProto nor in-memory bytes: hop through a file
    with tempfile.NamedTemporaryFile(suffix=".onnx") as handle:
        handle.write(onnx_bytes)
        handle.flush()
        input_scheme = [
            (
                name,
                [-1 if isinstance(dim, str) else dim for dim in manifest.input_shapes[name]],
            )
            for name in manifest.input_names
        ]
        ov_model = ov.convert_model(handle.name, input=input_scheme)
    return core.compile_model(ov_model, "CPU")


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().numpy()
    return np.asarray(value)


class CompiledModel:
    """CPU-optimized inference over an exported ``replay.nn`` graph.

    Instances are produced by :func:`compile` and :func:`load`;
    the graph contract (input names, shapes, candidates policy) lives in
    the :attr:`manifest`.
    """

    def __init__(self, compiled_model, manifest: ExportManifest, onnx_bytes: bytes):
        self._model = compiled_model
        self._manifest = manifest
        self._onnx_bytes = onnx_bytes

    @property
    def manifest(self) -> ExportManifest:
        """Export manifest describing the compiled graph."""
        return self._manifest

    @property
    def input_names(self) -> list[str]:
        """Names of the graph inputs, in schema order."""
        return list(self._manifest.input_names)

    def predict(self, inputs: dict[str, np.ndarray | torch.Tensor]) -> np.ndarray:
        """Run the compiled graph.

        :param inputs: arrays keyed by graph input name (see :attr:`input_names`);
            ``padding_mask`` and, under the explicit candidates policy,
            ``candidates`` are inputs like any other.
        :return: model logits as a NumPy array.
        """
        arrays = {name: _to_numpy(value) for name, value in inputs.items()}
        missing = [name for name in self._manifest.input_names if name not in arrays]
        unknown = [name for name in arrays if name not in self._manifest.input_names]
        if missing or unknown:
            msg = f"Predict inputs do not match the compiled graph: missing={missing}, unexpected={unknown}"
            raise ValueError(msg)
        result = self._model(arrays)
        return np.asarray(result[self._model.output(0)])

    def save(self, path: str | Path) -> None:
        """Persist the ONNX graph and its manifest into directory ``path``."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ONNX_FILENAME).write_bytes(self._onnx_bytes)
        (directory / MANIFEST_FILENAME).write_text(json.dumps(self._manifest.to_dict(), indent=2))

    def __repr__(self) -> str:
        return f"CompiledModel(model_type={self._manifest.model_type!r}, exporter={self._manifest.exporter!r})"


def compile(
    model: torch.nn.Module,
    schema,
    *,
    mode: str = MODE_DYNAMIC_BATCH,
    candidates_policy: str = CANDIDATES_FULL_VOCAB,
    num_candidates: int | None = None,
    batch_size: int = 2,
    max_seq_len: int = 32,
    exporter: str = AUTO,
    opset_version: int = DEFAULT_OPSET_VERSION,
    num_threads: int | None = None,
) -> CompiledModel:
    """Export a ``replay.nn`` model's ``forward_inference`` to ONNX and compile it with OpenVINO.

    :param model: a torch module from ``replay.nn`` (``SasRec``, ``TwoTower``).
    :param schema: the model's ``TensorSchema``; graph inputs follow its feature order.
    :param mode: ``dynamic_batch_size`` (batch and sequence length stay dynamic)
        or ``batch`` (batch is fixed at ``batch_size``, sequence length stays dynamic).
    :param candidates_policy: ``full_vocab`` or ``explicit`` (candidates become a graph input).
    :param num_candidates: candidate count to trace with; required for
        (and only valid with) the ``explicit`` policy.
    :param batch_size: batch size used for tracing; the fixed size under the ``batch`` mode.
    :param max_seq_len: sequence length used for tracing.
    :param exporter: ``auto`` (default; dynamo when the graph contains ``RMSNorm``,
        legacy otherwise), ``legacy`` or ``dynamo``.
    :param opset_version: ONNX opset version.
    :param num_threads: CPU threads for OpenVINO compilation; ``None`` lets OpenVino decide.
    :return: a :class:`CompiledModel` ready for :meth:`CompiledModel.predict`.
    """
    _check_conditional_stack()
    if mode not in _MODES:
        msg = f"Unknown mode: {mode!r}; expected one of {', '.join(_MODES)}"
        raise ValueError(msg)

    explicit = candidates_policy == CANDIDATES_EXPLICIT
    if explicit and num_candidates is None:
        msg = "num_candidates is required when candidates_policy is 'explicit'"
        raise ValueError(msg)
    if not explicit and num_candidates is not None:
        msg = "num_candidates is only accepted when candidates_policy is 'explicit'"
        raise ValueError(msg)

    adapter = resolve_adapter(model)
    wrapper, graph_inputs, features = adapter.build(model, schema, candidates_policy)
    wrapper.eval()

    sample = _build_sample(schema, graph_inputs, batch_size, max_seq_len, num_candidates)
    input_names = [graph_input.name for graph_input in graph_inputs]
    args = tuple(sample[name] for name in input_names)

    exporter_name = select_exporter(wrapper, exporter)
    onnx_bytes = export_onnx(
        wrapper,
        args,
        input_names,
        [adapter.output_name],
        _dynamic_axes(graph_inputs, mode, adapter.output_name),
        opset_version,
        exporter=exporter_name,
    )

    manifest = ExportManifest(
        model_type=adapter.model_type,
        exporter=exporter_name,
        opset_version=opset_version,
        input_kinds=[graph_input.kind for graph_input in graph_inputs],
        input_names=input_names,
        output_names=[adapter.output_name],
        candidates_policy=candidates_policy,
        features=features,
        input_shapes=_input_shapes(graph_inputs, sample, mode),
        versions=_stack_versions(),
    )
    compiled_model = _compile_openvino(onnx_bytes, manifest, num_threads)
    return CompiledModel(compiled_model=compiled_model, manifest=manifest, onnx_bytes=onnx_bytes)


def load(path: str | Path, *, num_threads: int | None = None) -> CompiledModel:
    """Load a saved artifact directory and compile it with OpenVino.

    :param path: directory holding ``model.onnx`` and ``manifest.json``
        (as written by :meth:`CompiledModel.save`).
    :param num_threads: CPU threads for OpenVINO compilation; ``None`` lets OpenVino decide.
    :return: a :class:`CompiledModel` ready for :meth:`CompiledModel.predict`.
    """
    _check_conditional_stack()
    directory = Path(path)
    manifest = ExportManifest.from_dict(json.loads((directory / MANIFEST_FILENAME).read_text()))
    onnx_bytes = (directory / ONNX_FILENAME).read_bytes()
    compiled_model = _compile_openvino(onnx_bytes, manifest, num_threads)
    return CompiledModel(compiled_model=compiled_model, manifest=manifest, onnx_bytes=onnx_bytes)
