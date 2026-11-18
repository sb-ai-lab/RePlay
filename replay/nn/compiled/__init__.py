"""ONNX and OpenVINO compilation of ``replay.nn`` inference graphs."""

from ._core import CompiledModel, compile, load

__all__ = ["CompiledModel", "compile", "load"]
