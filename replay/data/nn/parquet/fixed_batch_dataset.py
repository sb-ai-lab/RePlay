import warnings
from collections.abc import Iterator
from typing import Callable, Optional, Protocol, cast

import torch
from torch.utils.data import IterableDataset

from replay.constants.batches import GeneralBatch, GeneralCollateFn
from replay.data.nn.parquet.impl.masking import DEFAULT_COLLATE_FN


def get_batch_size(batch: GeneralBatch, strict: bool = False) -> int:
    """
    Получает размер батча из объекта `batch`.

    Аргументы:
        batch (GeneralBatch): Объект, содержащий батч.
        strict (bool, optional): Если `True`, то будет проведена дополнительная проверка.
        По умолчанию `False`.

    Возвращает:
        int: Размер батча.
    """
    batch_size: Optional[int] = None

    for key, value in batch.items():
        new_batch_size: int

        if torch.is_tensor(value):
            new_batch_size = value.size(0)
        else:
            assert isinstance(value, dict)
            new_batch_size = get_batch_size(value, strict)

        if batch_size is None:
            batch_size = new_batch_size

        if strict:
            if batch_size != new_batch_size:
                msg = f"Batch size mismatch {key}: {batch_size} != {new_batch_size}"
                raise ValueError(msg)
        else:
            break
    assert batch_size is not None
    return cast(int, batch_size)


def split_batches(batch: GeneralBatch, split: int) -> tuple[GeneralBatch, GeneralBatch]:
    left: GeneralBatch = {}
    right: GeneralBatch = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            sub_left = value[:split, ...]
            sub_right = value[split:, ...]
        else:
            sub_left, sub_right = split_batches(value, split)
        left[key], right[key] = sub_left, sub_right

    return (left, right)


class DatasetProtocol(Protocol):
    """
    Протокол для входного датасета.
    """

    def __iter__(self) -> Iterator[GeneralBatch]: ...
    @property
    def batch_size(self) -> int: ...


class FixedBatchSizeDataset(IterableDataset):
    """
    Обертка для произвольных датасетов. Позволяет получать батчи фиксированного размера.
        Конкатенирует батчи из обернутого датасета, пока не достигнет заданного размера.
        Последний батч может быть меньше заданного размера.

    Аргументы:
        dataset (DatasetProtocol): Итерируемый объект, возвращающий батчи.
            Чаще всего подкласс `torch.utils.data.IterableDataset`, например
            `fmlib.data.io.PartitionedIterableDataset`.
        batch_size (Optional[int]): Желанный размер батча. Если `None`, то будет
            предпринята попытка получить его из `dataset.batch_size`. По умолчанию `None`.
        collate_fn (GeneralCollateFn): Функция для объединения батчей. По умолчанию DEFAULT_COLLATE_FN.
        strict_checks (bool, optional): Если `True`, то будут проведены дополнительные проверки.
            Может влиять на скорость работы. По умолчанию `False`.

    Атрибуты:
        dataset (PartitionedIterableDataset): Объект датасета, поддерживающий итерацию по батчам.
    """

    def __init__(
        self,
        dataset: DatasetProtocol,
        batch_size: Optional[int] = None,
        collate_fn: GeneralCollateFn = DEFAULT_COLLATE_FN,
        strict_checks: bool = False,
    ) -> None:
        super().__init__()

        self.dataset: DatasetProtocol = dataset

        if batch_size is None:
            assert hasattr(dataset, "batch_size")
            batch_size = self.dataset.batch_size

        assert isinstance(batch_size, int)
        int_batch_size: int = cast(int, batch_size)

        if int_batch_size < 1:
            msg = f"Insufficient batch size. Got {int_batch_size=}"
            raise ValueError(msg)

        if int_batch_size < 2:
            warnings.warn(f"Low batch size. Got {int_batch_size=}. This may cause performance issues.", stacklevel=2)

        self.collate_fn: Callable = collate_fn
        self.batch_size: int = int_batch_size
        self.strict_checks: bool = strict_checks

    def get_batch_size(self, batch: GeneralBatch) -> int:
        return get_batch_size(batch, strict=self.strict_checks)

    def __iter__(self) -> Iterator[GeneralBatch]:
        iterator: Iterator[GeneralBatch] = iter(self.dataset)

        buffer: list[GeneralBatch] = []
        buffer_size: int = 0

        while True:
            while buffer_size < self.batch_size:
                try:
                    batch: GeneralBatch = next(iterator)
                    size: int = self.get_batch_size(batch)

                    buffer.append(batch)
                    buffer_size += size
                except StopIteration:
                    break

            if buffer_size == 0:
                break

            joined: GeneralBatch = self.collate_fn(buffer)
            assert buffer_size == self.get_batch_size(joined)

            if self.batch_size < buffer_size:
                left, right = split_batches(joined, self.batch_size)
                residue: int = buffer_size - self.batch_size
                assert residue == self.get_batch_size(right)

                buffer_size = residue
                buffer = [right]

                yield left
            else:
                buffer_size = 0
                buffer = []

                yield joined

        assert buffer_size == 0
        assert len(buffer) == 0
