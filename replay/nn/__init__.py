from replay.utils import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    msg = (
        "The replay.nn module is unavailable. "
        "To use the functionality from this module, please install ``torch`` and ``lightning``."
    )
    raise ImportError(msg)
