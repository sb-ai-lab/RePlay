from replay.utils import OPENVINO_AVAILABLE

if OPENVINO_AVAILABLE:
    from .bert4rec_compiled import Bert4RecCompiled
    from .sasrec_compiled import SasRecCompiled

    __all__ = ["Bert4RecCompiled", "SasRecCompiled"]
