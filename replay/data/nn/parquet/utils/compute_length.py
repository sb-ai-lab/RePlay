import warnings
from collections.abc import Iterable
from typing import Protocol

from replay.data.nn.parquet.info.partitioning import partitioning_per_replica
from replay.data.nn.parquet.iterator import BatchesIterator


class HasLengthProtocol(Protocol):
    def __len__(self) -> int: ...


def compute_fixed_size_generic_length_from_sizes(
    partition_sizes: Iterable[int], batch_size: int, num_replicas: int
) -> int:
    residue = 0
    batch_counter = 0
    for partition_size in partition_sizes:
        per_replica = partitioning_per_replica(partition_size, num_replicas)
        batch_count = per_replica // batch_size
        residue += per_replica % batch_size
        if batch_size < residue:
            batch_count += residue // batch_size
            residue = residue % batch_size
        batch_counter += batch_count
    batch_counter += residue > 0
    return batch_counter


def compute_fixed_size_batches_length(iterable: BatchesIterator, batch_size: int, num_replicas: int) -> int:
    assert isinstance(iterable, BatchesIterator)

    partition_size = iterable.batch_size

    def default_partitions(fragment_size: int) -> list[int]:
        full_partitions_count = fragment_size // partition_size
        result = [partition_size] * full_partitions_count
        if (residue := (fragment_size % partition_size)) > 0:
            result.append(residue)
        return result

    partition_sizes = []
    for fragment in iterable.dataset.get_fragments():
        fragment_size = fragment.count_rows()
        partitions = default_partitions(fragment_size)
        partition_sizes.extend(partitions)

    result = compute_fixed_size_generic_length_from_sizes(
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
