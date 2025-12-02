import pytest

from replay.data.parquet.iterable_dataset import validate_batch_size

def test_invalid_batch_size():
    with pytest.raises(ValueError):
        validate_batch_size(-1)