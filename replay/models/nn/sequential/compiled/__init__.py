try:
    import onnx
    import openvino

    from .sasrec_compiled import SasRecCompiled
except ImportError:  # pragma: no cover
    import warnings

    warnings.warn(
        "You are trying to import CPU-optimized model for inference via OpenVINO, but"
        "you have no onnx or openvino packages in your virtual env. Import will not be done."
    )
