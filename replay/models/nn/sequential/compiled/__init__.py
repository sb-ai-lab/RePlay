from replay.utils import OPENVINO_AVAILABLE

if OPENVINO_AVAILABLE:
    from .bert4rec_compiled import Bert4RecCompiled
    from .sasrec_compiled import SasRecCompiled

    __all__ = ["Bert4RecCompiled", "SasRecCompiled"]
else:
    import sys

    err = ImportError('Cannot import from module "compiled" - OpenVINO prerequisites not found.')
    if sys.version_info >= (3, 11):  # pragma: py-lt-311
        err.add_note('To enable this functionality, ensure you have both "openvino" and "onnx" packages isntalled.')

    raise err
