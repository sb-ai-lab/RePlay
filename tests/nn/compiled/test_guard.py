import pytest

pytest.importorskip("torch")

from replay.utils import OPENVINO_AVAILABLE

pytestmark = pytest.mark.skipif(OPENVINO_AVAILABLE, reason="onnx/openvino installed: guard branch unreachable")


def test_compile_raises_without_conditional_stack():
    from replay.nn.compiled import compile

    with pytest.raises(ImportError, match="openvino"):
        compile(None, None)


def test_load_raises_without_conditional_stack(tmp_path):
    from replay.nn.compiled import load

    with pytest.raises(ImportError, match="openvino"):
        load(tmp_path)
