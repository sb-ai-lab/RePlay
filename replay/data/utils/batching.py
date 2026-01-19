from functools import lru_cache
from typing import Iterator, Tuple


def validate_length(length: int) -> int:
    if length < 1:
        msg: str = f"Length is invalid. Got {length}."
        raise ValueError(msg)
    return length


def validate_batch_size(batch_size: int) -> int:
    if batch_size < 1:
        msg: str = f"Batch Size is invalid. Got {batch_size}."
        raise ValueError(msg)
    return batch_size


def validate_input(length: int, batch_size: int) -> Tuple[int, int]:
    length = validate_length(length)
    batch_size = validate_batch_size(batch_size)
    return (length, batch_size)


def uniform_batch_count(length: int, batch_size: int) -> int:
    @lru_cache
    def _uniform_batch_count(length: int, batch_size: int) -> int:
        length, batch_size = validate_input(length, batch_size)
        batch_count: int = length // batch_size
        batch_count = batch_count + bool(length % batch_size)
        assert batch_count >= 1
        assert length <= batch_count * batch_size
        assert (batch_count - 1) * batch_size < length
        return batch_count

    return _uniform_batch_count(length, batch_size)


class UniformBatching:
    def __init__(self, length: int, batch_size: int) -> None:
        length, batch_size = validate_input(length, batch_size)

        self.length: int = length
        self.batch_size: int = batch_size

    @property
    def batch_count(self) -> int:
        return uniform_batch_count(self.length, self.batch_size)

    def __len__(self) -> int:
        return self.batch_count

    def get_limits(self, index: int) -> Tuple[int, int]:
        if (index < 0) or (self.batch_count <= index):
            msg: str = f"Batching Index is invalid. Got {index}."
            raise IndexError(msg)
        first: int = index * self.batch_size
        last: int = min(self.length, first + self.batch_size)
        assert (first >= 0) and (first < self.length)
        assert (first < last) and (last <= self.length)
        return (first, last)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        return self.get_limits(index)

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        index: int
        for index in range(self.batch_count):
            yield self.get_limits(index)
