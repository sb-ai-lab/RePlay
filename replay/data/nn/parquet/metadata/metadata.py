from collections.abc import Callable
from typing import Any, Union

from typing_extensions import TypeAlias

from replay.data.nn.parquet.constants.metadata import (
    DEFAULT_PADDING,
    PADDING_FLAG,
    SHAPE_FLAG,
)

FieldType: TypeAlias = Union[bool, int, float, str]
ColumnMetadata: TypeAlias = dict[str, FieldType]
Metadata: TypeAlias = dict[str, ColumnMetadata]

ColumnCheck: TypeAlias = Callable[[ColumnMetadata], bool]
CheckColumn: TypeAlias = Callable[[ColumnCheck], bool]
Listing: TypeAlias = Callable[[Metadata], list[str]]


def make_shape_check(dim: int) -> ColumnCheck:
    """
    Constructs a function which checks a column's shape.

    :param dim: Target number of dimensions.
    """

    def function(column_metadata: ColumnMetadata) -> bool:
        if SHAPE_FLAG in column_metadata:
            value: Any = column_metadata[SHAPE_FLAG]
            if dim == 1 and isinstance(value, int):
                return True
            if isinstance(value, list):
                result: bool = len(value) == dim
                if result:

                    def is_int(v: Any) -> bool:
                        return isinstance(v, int)

                    result &= all(map(is_int, value))
                return result
        return False

    return function


def make_not_check(check: ColumnCheck) -> ColumnCheck:
    def function(column_metadata: ColumnCheck) -> bool:
        return not check(column_metadata)

    return function


def all_column_checks(*checks: ColumnCheck) -> ColumnCheck:
    def function(column_metadata: ColumnMetadata) -> bool:
        def perform_check(check):
            return check(column_metadata)

        return all(map(perform_check, checks))

    return function


is_array_1d = all_column_checks(make_shape_check(dim=1))
is_array_2d = all_column_checks(make_shape_check(dim=2))
is_number = all_column_checks(
    make_not_check(is_array_1d),
    make_not_check(is_array_2d),
)


def make_listing(check: ColumnCheck) -> Listing:
    """
    Filtering function for selecting columns that pass the provided check.

    :param check: Check function to validate agains.
    """

    def function(metadata: Metadata) -> list[str]:
        result: list[str] = []
        for col_name, col_meta in metadata.items():
            if check(col_meta):
                result.append(col_name)
        return sorted(result)

    return function


get_1d_array_columns = make_listing(is_array_1d)
get_2d_array_columns = make_listing(is_array_2d)
get_numeric_columns = make_listing(is_number)


def get_padding(metadata: Metadata, column_name: str) -> Any:
    if column_name not in metadata:
        msg = f"Column {column_name} not found in metadata."
        raise KeyError(msg)
    return metadata[column_name].get(PADDING_FLAG, DEFAULT_PADDING)


def get_shape(metadata: Metadata, column_name: str) -> list[int]:
    if column_name not in metadata:
        msg = f"Column {column_name} not found in metadata."
        raise KeyError(msg)
    if is_number(metadata[column_name]):
        msg = f"Column {column_name} is not an array."
        raise ValueError(msg)
    result: Any = metadata[column_name][SHAPE_FLAG]

    array_res: list[Any] = result if isinstance(result, list) else [result]

    for i in range(len(array_res)):
        if array_res[i] < 1:
            msg = f"Shape for column {column_name} at position {i} is not a positive integer."
            raise ValueError(msg)
    return result
