import pytest
import torch

from replay.data.nn.parquet.info.partitioning import (
    Partitioning,
    partitioning_length,
    partitioning_per_replica,
)


def test_partitioning_length():
    assert partitioning_length(1, 3) == (1 * 3)
    assert partitioning_length(111, 4) == (28 * 4)
    assert partitioning_length(111, 3) == (37 * 3)
    assert partitioning_length(111, 1) == (111 * 1)


def test_partitioning_wrong_length():
    with pytest.raises(ValueError):
        _ = partitioning_length(3, -1)
    with pytest.raises(ValueError):
        _ = partitioning_length(-1, 3)
    with pytest.raises(ValueError):
        _ = partitioning_length(-2, -3)


def test_partitioning_per_replica():
    assert partitioning_per_replica(1, 3) == 1
    assert partitioning_per_replica(111, 4) == 28
    assert partitioning_per_replica(111, 3) == 37
    assert partitioning_per_replica(111, 1) == 111


def test_partitioning_wrong_per_replica():
    with pytest.raises(ValueError):
        _ = partitioning_per_replica(3, -1)
    with pytest.raises(ValueError):
        _ = partitioning_per_replica(-1, 3)
    with pytest.raises(ValueError):
        _ = partitioning_per_replica(-2, -3)


def test_partitioning_sequential():
    assert (
        Partitioning(
            curr_replica=1,
            num_replicas=3,
        )
        .generate(5)
        .cpu()
        .tolist()
    ) == [1, 4]

    assert (
        Partitioning(
            curr_replica=2,
            num_replicas=3,
        )
        .generate(5)
        .cpu()
        .tolist()
    ) == [2, 0]

    assert (
        Partitioning(
            curr_replica=0,
            num_replicas=1,
        )
        .generate(3)
        .cpu()
        .tolist()
    ) == [0, 1, 2]

    with pytest.raises(ValueError):
        Partitioning(
            curr_replica=-1,
            num_replicas=1,
        )


def count_occurances(items: list[int]) -> dict[int, int]:
    count_dict: dict[int, int] = {}
    element: int
    for element in items:
        if element in count_dict:
            count_dict[element] += 1
        else:
            count_dict[element] = 1
    return count_dict


@pytest.mark.parametrize("num_replicas", [1, 2, 3, 4])
@pytest.mark.parametrize("length", [1, 2, 3, 8, 11, 17])
def test_partitioning_sequential_automated(num_replicas: int, length: int):
    indices: list[int] = []
    for curr_replica in range(num_replicas):
        slice_indices: list[int] = (
            Partitioning(
                curr_replica=curr_replica,
                num_replicas=num_replicas,
            )
            .generate(length)
            .cpu()
            .tolist()
        )
        indices = indices + slice_indices
    counts: dict[int, int] = count_occurances(indices)
    threshold: int = 2 if num_replicas <= length else num_replicas
    assert all((count <= threshold) for count in counts.values())
    assert all((count >= 1) for count in counts.values())


@pytest.mark.parametrize("seed", [1, 42, 777])
@pytest.mark.parametrize("num_replicas", [1, 2, 3, 4])
@pytest.mark.parametrize("length", [1, 2, 3, 8, 11, 17])
def test_partitioning_random_automated(seed: int, num_replicas: int, length: int):
    indices: list[int] = []
    for curr_replica in range(num_replicas):
        generator: torch.Generator = torch.Generator().manual_seed(seed)
        slice_indices: list[int] = (
            Partitioning(
                generator=generator,
                curr_replica=curr_replica,
                num_replicas=num_replicas,
            )
            .generate(length)
            .cpu()
            .tolist()
        )
        indices = indices + slice_indices
    counts: dict[int, int] = count_occurances(indices)
    threshold: int = 2 if num_replicas <= length else num_replicas
    assert all((count <= threshold) for count in counts.values())
    assert all((count >= 1) for count in counts.values())
