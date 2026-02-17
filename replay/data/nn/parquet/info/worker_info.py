from collections.abc import Iterator
from typing import Any, Protocol

import torch.utils.data as data


class WorkerInfoProtocol(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def num_workers(self) -> int: ...


class WorkerInfo:
    """Wrapper class for Torch's worker metadata."""

    def __iter__(self) -> Iterator[int]:
        yield self.id

    @property
    def worker_info(self) -> Any | None:
        return data.get_worker_info()

    @property
    def is_parallel(self) -> bool:
        return self.worker_info is not None

    @property
    def id(self) -> int:
        wi: data.WorkerInfo | None = self.worker_info
        if wi is not None:
            return wi.id
        return 0

    @property
    def num_workers(self) -> int:
        wi: data.WorkerInfo | None = self.worker_info
        if wi is not None:
            return wi.num_workers
        return 1


DEFAULT_WORKER_INFO: WorkerInfo = WorkerInfo()
