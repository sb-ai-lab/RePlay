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
    """
    A replica metadata geneartor.

    By default, assumes standard Torch DDP training/inference procedure,
    where each replica (a distinct worker on a specific device) is expected to process
    a separate chunk of the dataset.

    This behavior can be modified by providing custom ``worker_info`` and ``distributed_info`` objects
    able to provide infor about local worker count and world size/rank respectively.
    """

    def __init__(
        self,
        worker_info: WorkerInfoProtocol = DEFAULT_WORKER_INFO,
        distributed_info: DistributedInfoProtocol = DEFAULT_DISTRIBUTED_INFO,
    ) -> None:
        """
        :param worker_info: An object adhering to the ``WorkerInfoProtocol`` and used to obtain local worker count.
            Default: value of ``DEFAULT_WORKER_INFO`` - an implementation using ``torch.utils.data.get_worker_info()``.
        :param distributed_info: An object adhering to the ``DistributedInfoProtocol`` and used to obtain
            world size and rank. Default: value of ``DEFAULT_WORKER_INFO`` - an implementation using the
            ``torch.distributed`` module.
        """
        self.worker_info = worker_info
        self.distributed_info = distributed_info

    @property
    def num_replicas(self) -> int:
        return num_replicas(worker_info=self.worker_info, distributed_info=self.distributed_info)

    @property
    def curr_replica(self) -> int:
        return curr_replica(worker_info=self.worker_info, distributed_info=self.distributed_info)


DEFAULT_REPLICAS_INFO: ReplicasInfoProtocol = ReplicasInfo()
