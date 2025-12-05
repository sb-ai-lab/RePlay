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

    Arguments:
        named_columns (NamedColumns): Структурированные данные, представленные в виде колонок.
        batch_size (int): Размер одного батча.
        generator (Optional[torch.Generator], optional): Генератор случайных чисел для перемешивания батчей.
            Если не указан, то перемешивание будет выключено.
        replicas_info (ReplicasInfoProtocol, optional): Информация о репликах. По умолчанию DEFAULT_REPLICAS_INFO.

    Attributes:
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

        self.named_columns = named_columns
        self.generator = generator
        self.replicas_info = replicas_info
        self.batch_size = validate_batch_size(batch_size)

    @property
    def device(self) -> torch.device:
        """Returns the device containing the dataset."""
        return self.named_columns.device

    @property
    def full_length(self) -> int:
        """Returns the total amount of elements in `named_columns`."""
        return self.named_columns.length

    @property
    def length_per_replica(self) -> int:
        """Returns the total number of available elements per replica."""
        full_length = self.named_columns.length
        num_replicas = self.replicas_info.num_replicas
        return partitioning_per_replica(full_length, num_replicas)

    @property
    def length(self) -> int:
        """Returns the total number of batches available to the current replica."""
        batch_size = self.batch_size
        per_replica = self.length_per_replica
        return uniform_batch_count(per_replica, batch_size)

    def __len__(self) -> int:
        """Returns the total number of batches in a dataset."""
        return self.length

    def get_indices(self) -> torch.LongTensor:
        """
        Generates indices corresponding to data assigned to current replica.

        :return: tensor containing relevant indices.
        """
        partitioning = Partitioning(
            curr_replica=self.replicas_info.curr_replica,
            num_replicas=self.replicas_info.num_replicas,
            device=self.named_columns.device,
            generator=self.generator,
        )
        indices = partitioning(self.full_length)
        assert self.length_per_replica == torch.numel(indices)
        return indices

    def get_batching(self) -> UniformBatching:
        """
        Creates a partitioning object which splits data into batches.

        :return: The partitioning object.
        """
        batching = UniformBatching(
            length=self.length_per_replica,
            batch_size=self.batch_size,
        )
        assert len(batching) == self.length
        return batching

    def __iter__(self) -> Iterator[Batch]:
        """Batched data iterator."""
        batching = self.get_batching()
        indices = self.get_indices()

        for first, last in iter(batching):
            batch_ids = indices[first:last]
            yield self.named_columns[batch_ids]
