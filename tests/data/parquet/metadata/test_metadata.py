import pytest

from replay.constants.metadata import SHAPE_FLAG
from replay.data.parquet.metadata import (
    get_1d_array_columns,
    get_2d_array_columns,
    get_numeric_columns,
    get_padding,
    get_shape,
)

TEST_METADATA = {
    "ft_0": {},
    "seq_0": {SHAPE_FLAG: [3]},
    "seq_1": {SHAPE_FLAG: [1, 2]}
}


def test_1d():
    result = get_1d_array_columns(TEST_METADATA)
    assert result == ["seq_0"]

def test_2d():
    result = get_2d_array_columns(TEST_METADATA)
    assert result == ["seq_1"]

def test_numeric():
    result = get_numeric_columns(TEST_METADATA)
    assert result == ["ft_0"]


def test_padding_nonexistent_column():
    with pytest.raises(KeyError):
        get_padding(TEST_METADATA, "nihao")


def test_shape_nonexistent_column():
    with pytest.raises(KeyError):
        get_shape(TEST_METADATA, "nihao")

def test_shape_on_numeric_column() -> None:
    with pytest.raises(ValueError):
        get_shape(TEST_METADATA, "ft_0")

def test_shape_malformed_meta() -> None:
    with pytest.raises(ValueError):
        get_shape({"seq": {SHAPE_FLAG: [1, -2]}}, "seq")
