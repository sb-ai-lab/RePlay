try:
    import onnx
    import openvino

    from .sasrec_compiled import SasRecCompiled
except ImportError:
    # warning
    pass
