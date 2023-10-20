""" RecSys library """
try:
    from . import experimental

    _EXPERIMENTAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _EXPERIMENTAL_AVAILABLE = False

__version__ = "0.0.0"

__all__ = ["data", "metrics", "models", "optimization", "preprocessing", "scenarios", "splitters", "utils"] + (
    ["experimental"] if _EXPERIMENTAL_AVAILABLE else []  # pragma: no cover
)
