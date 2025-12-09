from functools import lru_cache
from math import ceil
from typing import Optional, Union

import torch

from replay.data.nn.parquet.constants.device import DEFAULT_DEVICE


def validate_length(length: int) -> int:
    if length < 1:
        msg = f"Length is invalid. Got {length}."
        raise ValueError(msg)
    return length


def validate_num_replicas(num_replicas: int) -> int:
    if num_replicas < 1:
        msg = f"Num Replicas is invalid. Got {num_replicas}."
        raise ValueError(msg)
    return num_replicas


def validate_curr_replica(curr_replica: int, num_replicas: int) -> int:
    num_replicas = validate_num_replicas(num_replicas)
    if (curr_replica < 0) or (num_replicas <= curr_replica):
        msg = f"Curr Replicas is invalid. Got {curr_replica}."
        raise ValueError(msg)
    return curr_replica


@lru_cache
def _partitioning_length(length: int, num_replicas: int) -> int:
    length = validate_length(length)
    num_replicas = validate_num_replicas(num_replicas)

    result = length
    if length % num_replicas != 0:
        raw_per_replica = length / num_replicas
        per_replica = ceil(raw_per_replica)
        new_length = per_replica * num_replicas
        assert (new_length - length) < num_replicas
        result = new_length
    assert result % num_replicas == 0
    assert length <= result
    return result


def partitioning_length(length: int, num_replicas: int) -> int:
    return _partitioning_length(length, num_replicas)


@lru_cache
def _partitioning_per_replica(length: int, num_replicas: int) -> int:
    full_length = partitioning_length(length, num_replicas)
    result = full_length // num_replicas
    assert result <= length
    assert result > 0
    return result


def partitioning_per_replica(length: int, num_replicas: int) -> int:
    return _partitioning_per_replica(length, num_replicas)


class Partitioning:
    """Utility class for calculating valid indices across multiple replicas."""

    def __init__(
        self,
        curr_replica: int,
        num_replicas: int,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        :param curr_replica: Id of the curreent replica.
        :param num_replicas: Total number of active replicas.
        :param device: Target device to send the indices tensor to.
            Default: value of ``DEFAULT_DEVICE``.
        :param generator: A pseudo-random number generator for index shuffling. Default: ``None``.
        """
        self.device = torch.device(device)
        self.generator = generator
        self.num_replicas = validate_num_replicas(num_replicas)
        self.curr_replica = validate_curr_replica(curr_replica, self.num_replicas)

    def generate_raw_indices(self, length: int) -> torch.LongTensor:
        full_length = partitioning_length(length, self.num_replicas)

        if self.generator is None:
            raw_indices = torch.arange(full_length, dtype=torch.int64, device=self.device)
        else:
            raw_indices = torch.randperm(full_length, dtype=torch.int64, generator=self.generator)
            raw_indices = raw_indices.to(device=self.device)

        assert torch.max(raw_indices).cpu().item() < full_length
        assert torch.numel(raw_indices) == full_length
        assert raw_indices.device == self.device

        return raw_indices

    def replica_indices(self, raw_indices: torch.LongTensor) -> torch.LongTensor:
        full_length = torch.numel(raw_indices)
        slc = slice(self.curr_replica, full_length, self.num_replicas)
        replica_indices = raw_indices[slc].clone()

        assert torch.max(replica_indices).cpu().item() < full_length

        return replica_indices

    def generate(self, length: int) -> torch.LongTensor:
        raw_indices = self.generate_raw_indices(length)
        full_length = partitioning_length(length, self.num_replicas)

        assert torch.numel(raw_indices) == full_length

        replica_indices = self.replica_indices(raw_indices)
        per_replica = partitioning_per_replica(length, self.num_replicas)

        assert torch.numel(replica_indices) == per_replica

        indices = torch.remainder(replica_indices, length)

        assert torch.max(indices).cpu().item() < length
        assert torch.numel(indices) == per_replica
        assert indices.device == self.device

        return indices

    def __call__(self, length: int) -> torch.LongTensor:
        return self.generate(length)
