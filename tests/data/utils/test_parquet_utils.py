import pytest

from replay.data.utils.batching import UniformBatching, validate_input

pytest.importorskip("torch", reason="Module 'torch' is required for ParquetDataset tests.")


@pytest.mark.parametrize("length, batch_size", [(-1, 1), (1, -1)])
def test_validation_failures(length: int, batch_size: int) -> None:
    with pytest.raises(ValueError):
        validate_input(length, batch_size)


def test_uniform_batching() -> None:
    length, batch_size = 8, 4
    batching = UniformBatching(length, batch_size)

    with pytest.raises(IndexError) as exc:
        batching[9]
    assert "Batching Index is invalid." in str(exc.value)
