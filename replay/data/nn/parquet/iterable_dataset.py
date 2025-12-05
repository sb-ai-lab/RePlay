from collections.abc import Iterator
from typing import Optional

import torch
import torch.utils.data as data

from replay.data.nn.parquet.impl.masking import DEFAULT_REPLICAS_INFO
from replay.data.utils.batching import UniformBatching, uniform_batch_count

from .impl.named_columns import NamedColumns
from .info.partitioning import Partitioning, partitioning_per_replica
from .info.replicas import ReplicasInfoProtocol

Batch = dict[str, torch.Tensor]


def validate_batch_size(batch_size: int) -> int:
    if batch_size <= 0:
        msg = "`batch_size` must be a positive integer."
        raise ValueError(msg)
    return batch_size


class IterableDataset(data.IterableDataset):
    """
    Реализация итерируемого датасета, поддерживающего партиционирование и батчевание данных.
    Есть поддержка распределённого обучения, когда данные должны быть
        разделены между репликами, и воспроизводимое случайное перемешивание.
    Реплика - это идентификатор процесса, который выполняет загрузку данных.
        Определяется двумя параметрами - номер процесса внутри PyTorch и номер ноды.

    Аргументы:
        named_columns (NamedColumns): Структурированные данные, представленные в виде колонок.
        batch_size (int): Размер одного батча.
        generator (Optional[torch.Generator], optional): Генератор случайных чисел для перемешивания батчей.
            Если не указан, то перемешивание будет выключено.
        replicas_info (ReplicasInfoProtocol, optional): Информация о репликах. По умолчанию DEFAULT_REPLICAS_INFO.

    Атрибуты:
        named_columns (NamedColumns): Входные данные.
        generator (Optional[torch.Generator]): Генератор случайных чисел.
        replicas_info (ReplicasInfoProtocol): Информация о репликах.
        batch_size (int): Размер батча.
    """

    def __init__(
        self,
        named_columns: NamedColumns,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        replicas_info: ReplicasInfoProtocol = DEFAULT_REPLICAS_INFO,
    ) -> None:
        super().__init__()

        self.named_columns: NamedColumns = named_columns
        self.generator: Optional[torch.Generator] = generator
        self.replicas_info: ReplicasInfoProtocol = replicas_info
        self.batch_size: int = validate_batch_size(batch_size)

    @property
    def device(self) -> torch.device:
        """Возвращает устройство, на котором находятся данные."""
        return self.named_columns.device

    @property
    def full_length(self) -> int:
        """Возвращает общее количество элементов в `named_columns`."""
        return self.named_columns.length

    @property
    def length_per_replica(self) -> int:
        """Возвращает количество элементов, отведенное одной реплике."""
        full_length: int = self.named_columns.length
        num_replicas: int = self.replicas_info.num_replicas
        return partitioning_per_replica(full_length, num_replicas)

    @property
    def length(self) -> int:
        """Возвращает общее количество батчей, доступных для текущей реплики."""
        batch_size: int = self.batch_size
        per_replica: int = self.length_per_replica
        return uniform_batch_count(per_replica, batch_size)

    def __len__(self) -> int:
        """Возвращает количество батчей в датасете."""
        return self.length

    def get_indices(self) -> torch.LongTensor:
        """
        Генерирует индексы, соответствующие данным, назначенные текущей реплике.

        Returns:
            torch.LongTensor: Тензор с индексами.
        """
        partitioning: Partitioning = Partitioning(
            curr_replica=self.replicas_info.curr_replica,
            num_replicas=self.replicas_info.num_replicas,
            device=self.named_columns.device,
            generator=self.generator,
        )
        indices: torch.LongTensor = partitioning(self.full_length)
        assert self.length_per_replica == torch.numel(indices)
        return indices

    def get_batching(self) -> UniformBatching:
        """
        Создает объект партиционирования, разбивающий данные на батчи.

        Returns:
            UniformBatching: Объект партиционирования.
        """
        batching: UniformBatching = UniformBatching(
            length=self.length_per_replica,
            batch_size=self.batch_size,
        )
        assert len(batching) == self.length
        return batching

    def __iter__(self) -> Iterator[Batch]:
        """
        Итератор по батчам данных.

        Returns:
            Iterator[Batch]: Итератор, возвращающий батчи.
        """
        batching: UniformBatching = self.get_batching()
        indices: torch.LongTensor = self.get_indices()

        first: int
        last: int
        for first, last in iter(batching):
            batch_ids: torch.LongTensor = indices[first:last]
            yield self.named_columns[batch_ids]
