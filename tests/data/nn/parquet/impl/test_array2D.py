import pytest
import torch

from replay.data.nn.parquet.impl.array_2d_column import Array2DColumn


def test_invalid_shape():
    INVALID_LENGTHS_SHAPE = [1]

    with pytest.raises(ValueError) as exc:
        _ = Array2DColumn([], torch.tensor([0]), torch.tensor([0]), INVALID_LENGTHS_SHAPE, padding=-1)
    assert "Array2DColumn accepts a shape of size (2,)" in str(exc.value)
