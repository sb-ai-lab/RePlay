import functools
import inspect
import warnings
from typing import Any, Callable, Optional

from replay.utils import (
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
)


def _check_if_dataframe(var: Any):
    if not isinstance(var, (SparkDataFrame, PolarsDataFrame, PandasDataFrame)):
        msg = f"Object of type {type(var)} is not a dataframe of known type (can be pandas|spark|polars)"
        raise ValueError(msg)


def check_if_dataframe(*args_to_check: str) -> Callable[..., Any]:
    def decorator_func(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrap_func(*args: Any, **kwargs: Any) -> Any:
            extended_kwargs = {}
            extended_kwargs.update(kwargs)
            extended_kwargs.update(dict(zip(inspect.signature(func).parameters.keys(), args)))
            # add default param values to dict with arguments
            extended_kwargs.update(
                {
                    x.name: x.default
                    for x in inspect.signature(func).parameters.values()
                    if x.name not in extended_kwargs and x.default is not x.empty
                }
            )
            vals_to_check = [extended_kwargs[_arg] for _arg in args_to_check]
            for val in vals_to_check:
                _check_if_dataframe(val)
            return func(*args, **kwargs)

        return wrap_func

    return decorator_func


def deprecation_warning(message: Optional[str] = None) -> Callable[..., Any]:
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
