from collections.abc import Iterable, Iterator
from typing import Optional

import torch
import torch.utils.data as data

from replay.data.nn.parquet.impl.masking import DEFAULT_REPLICAS_INFO

from .impl.named_columns import NamedColumns
from .info.replicas import ReplicasInfoProtocol
from .iterable_dataset import IterableDataset

Batch = dict[str, torch.Tensor]


class PartitionedIterableDataset(data.IterableDataset):
    """
    Датасет, реализующий итерацию по данным, разбитым на партиции (partitions),
    с последующим формированием батчей для обучения модели.

    Эта реализация позволяет обрабатывать большие объемы данных в режиме потока,
    особенно полезна при использовании в распределённых тренировках.

    Аргументы:
        iterable (Iterable[NamedColumns]): Итерируемый объект, возвращающий партиции данных.
        batch_size (int): Размер одного батча.
        generator (Optional[torch.Generator], optional): Генератор случайных чисел для перемешивания батчей.
            Если не указан, то перемешивание будет выключено.
        replicas_info (ReplicasInfoProtocol, optional): Информация о репликах, используется в распределённой тренировке.
            По умолчанию — DEFAULT_REPLICAS_INFO.

    Атрибуты:
        iterable (Iterable[NamedColumns]): Итерируемый объект с партициями.
        batch_size (Optional[int]): Размер батча.
        generator (Optional[torch.Generator]): Генератор случайных чисел.
        replicas_info (ReplicasInfoProtocol): Информация о репликах.
    """

    def __init__(
        self,
        iterable: Iterable[NamedColumns],
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        replicas_info: ReplicasInfoProtocol = DEFAULT_REPLICAS_INFO,
    ) -> None:
        super().__init__()

        self.iterable = iterable

        self.batch_size = batch_size
        self.generator = generator
        self.replicas_info = replicas_info

    def __iter__(self) -> Iterator[Batch]:
        for partition in iter(self.iterable):
            iterable = IterableDataset(
                named_columns=partition,
                generator=self.generator,
                batch_size=self.batch_size,
                replicas_info=self.replicas_info,
            )

            yield from iter(iterable)
