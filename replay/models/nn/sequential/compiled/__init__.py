from replay.utils import OPENVINO_AVAILABLE

if OPENVINO_AVAILABLE:
    from .sasrec_compiled import SasRecCompiled
