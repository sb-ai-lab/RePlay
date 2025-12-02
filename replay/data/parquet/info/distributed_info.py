from typing import Protocol

import torch.distributed as dist


class DistributedInfo:
    def __iter__(self):
        yield self.rank
        yield self.world_size

    @property
    def is_distributed(self) -> bool:
        if dist.is_available():
            return dist.is_initialized()
        return False

    @property
    def rank(self) -> int:
        if self.is_distributed:
            return dist.get_rank()
        return 0

    @property
    def world_size(self) -> int:
        if self.is_distributed:
            return dist.get_world_size()
        return 1


class DistributedInfoProtocol(Protocol):
    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...


DEFAULT_DISTRIBUTED_INFO: DistributedInfo = DistributedInfo()
