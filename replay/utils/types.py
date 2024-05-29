from typing import Iterable, Union, Callable, Any
import functools
import warnings

from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame


class MissingImportType:
    """
    Replacement class with missing import
    """


try:
    from pyspark.sql import DataFrame as SparkDataFrame

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = MissingImportType

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DataFrameLike = Union[PandasDataFrame, SparkDataFrame, PolarsDataFrame]
IntOrList = Union[Iterable[int], int]
NumType = Union[int, float]


def deprecation_warning(message: str = None) -> Callable[..., Any]:
    """
    Decorator that throws deprecation warnings.

    :param message: message to deprecation warning without func name.
    """

    base_msg = "will be deprecated in future versions."

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            msg = f"{func.__qualname__} {message if message else base_msg}"
            warnings.simplefilter("always", DeprecationWarning)  # turn off filter
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter("default", DeprecationWarning)  # reset filter
            return func(*args, **kwargs)

        return wrapper

    return decorator
