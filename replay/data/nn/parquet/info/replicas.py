from typing import Protocol

from .distributed_info import DEFAULT_DISTRIBUTED_INFO, DistributedInfoProtocol
from .worker_info import DEFAULT_WORKER_INFO, WorkerInfoProtocol


def num_replicas(
    worker_info: WorkerInfoProtocol = DEFAULT_WORKER_INFO,
    distributed_info: DistributedInfoProtocol = DEFAULT_DISTRIBUTED_INFO,
) -> int:
    return worker_info.num_workers * distributed_info.world_size


def curr_replica(
    worker_info: WorkerInfoProtocol = DEFAULT_WORKER_INFO,
    distributed_info: DistributedInfoProtocol = DEFAULT_DISTRIBUTED_INFO,
) -> int:
    result = worker_info.id + worker_info.num_workers * distributed_info.rank
    assert result < num_replicas(worker_info, distributed_info)
    return result


class ReplicasInfoProtocol(Protocol):
    @property
    def num_replicas(self) -> int: ...

    @property
    def curr_replica(self) -> int: ...


class ReplicasInfo:
    """Wrapper class for Torch's replica metadata."""

    def __init__(
        self,
        worker_info: WorkerInfoProtocol = DEFAULT_WORKER_INFO,
        distributed_info: DistributedInfoProtocol = DEFAULT_DISTRIBUTED_INFO,
    ) -> None:
        self.worker_info = worker_info
        self.distributed_info = distributed_info

    @property
    def num_replicas(self) -> int:
        return num_replicas(worker_info=self.worker_info, distributed_info=self.distributed_info)

    @property
    def curr_replica(self) -> int:
        return curr_replica(worker_info=self.worker_info, distributed_info=self.distributed_info)
