import functools
import warnings
from collections.abc import Callable
from typing import Any, Optional


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
