import warnings
from collections.abc import Iterable
from typing import Protocol

from replay.data.parquet.iterator import BatchesIterator
from replay.data.parquet.info.partitioning import partitioning_per_replica


class HasLengthProtocol(Protocol):
    def __len__(self) -> int: ...


def compute_fixed_size_generic_length_from_sizes(
    partition_sizes: Iterable[int], batch_size: int, num_replicas: int
) -> int:
    residue: int = 0
    batch_counter: int = 0
    for partition_size in partition_sizes:
        per_replica: int = partitioning_per_replica(partition_size, num_replicas)
        batch_count: int = per_replica // batch_size
        residue += per_replica % batch_size
        if batch_size < residue:
            batch_count += residue // batch_size
            residue = residue % batch_size
        batch_counter += batch_count
    batch_counter += residue > 0
    return batch_counter


def compute_fixed_size_batches_length(iterable: BatchesIterator, batch_size: int, num_replicas: int) -> int:
    assert isinstance(iterable, BatchesIterator)

    partition_size: int = iterable.batch_size

    def default_partitions(fragment_size: int) -> list[int]:
        full_partitions_count: int = fragment_size // partition_size
        result: list[int] = [partition_size] * full_partitions_count
        if (residue := (fragment_size % partition_size)) > 0:
            result.append(residue)
        return result

    partition_sizes: list[int] = []
    for fragment in iterable.dataset.get_fragments():
        fragment_size: int = fragment.count_rows()
        partitions: list[int] = default_partitions(fragment_size)
        partition_sizes.extend(partitions)

    result: int = compute_fixed_size_generic_length_from_sizes(
        partition_sizes=partition_sizes,
        num_replicas=num_replicas,
        batch_size=batch_size,
    )

    return result


def compute_fixed_size_generic_length(iterable: Iterable[HasLengthProtocol], batch_size: int, num_replicas: int) -> int:
    warnings.warn("Generic length computation. This may cause performance issues.", UserWarning, stacklevel=2)
    return compute_fixed_size_generic_length_from_sizes(map(len, iterable), batch_size, num_replicas)


def compute_fixed_size_length(iterable: Iterable[HasLengthProtocol], batch_size: int, num_replicas: int) -> int:
    if isinstance(iterable, BatchesIterator):
        return compute_fixed_size_batches_length(iterable, batch_size, num_replicas)
    else:
        return compute_fixed_size_generic_length(iterable, batch_size, num_replicas)
