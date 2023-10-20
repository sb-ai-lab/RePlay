""" RecSys library """
from . import data, metrics, models, optimization, preprocessing, scenarios, splitters, utils

try:
    from . import experimental

    experimental_available = True
except ImportError:  # pragma: no cover
    experimental_available = False

__version__ = "0.0.0"

__all__ = ["data", "metrics", "models", "optimization", "preprocessing", "scenarios", "splitters", "utils"] + (
    ["experimental"] if experimental_available else []  # pragma: no cover
)
