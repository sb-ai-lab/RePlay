from collections.abc import Iterable, Iterator
from typing import Optional

import torch
import torch.utils.data as data

from replay.data.nn.parquet import DEFAULT_REPLICAS_INFO

from .impl.named_columns import NamedColumns
from .info.replicas import ReplicasInfoProtocol
from .iterable_dataset import IterableDataset

Batch = dict[str, torch.Tensor]


class PartitionedIterableDataset(data.IterableDataset):
    """
    A dataset that implements iteration over partitioned data.

    This implementation allows large amounts of data to be processed in batch-wise mode,
    which is especially useful when used in distributed training.
    """

    def __init__(
        self,
        iterable: Iterable[NamedColumns],
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        replicas_info: ReplicasInfoProtocol = DEFAULT_REPLICAS_INFO,
    ) -> None:
        """
        :param iterable: An iterable object that returns data partitions.
        :param batch_size: Batch size.
        :param generator: Random number generator for batch shuffling.
            If ``None``, shuffling will be disabled. Default: ``None``.
        :param replicas_info: A connector object capable of fetching total replica count and replica id during runtime.
            Default: value of ``DEFAULT_REPLICAS_INFO`` - a pre-built connector which assumes standard Torch DDP mode.
        """
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
