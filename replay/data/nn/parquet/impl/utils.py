import numpy as np

WRITEABLE_FLAG: str = "WRITEABLE"


def ensure_mutable(array: np.array) -> np.array:
    """
    Ensures the resulting NumPy array is mutable by making a copy if it's not.

    :param array: Array to be checked for mutability.
    :return: Mutable copy of `array`.
    """
    if not array.flags[WRITEABLE_FLAG]:
        result = array.copy()
        assert result.flags[WRITEABLE_FLAG]
        return result
    return array
