try:
    import onnx
    import openvino

    from .bert4rec_compiled import Bert4RecCompiled
except ImportError:
    # warning
    pass
