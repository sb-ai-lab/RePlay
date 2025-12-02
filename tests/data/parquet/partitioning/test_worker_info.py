from dataclasses import dataclass

import pytest as _

from replay.data.parquet.info.worker_info import DEFAULT_WORKER_INFO as worker_info

@dataclass
class MockWorker:
    id: int
    num_workers: int

def test_distributed_worker(mocker):
    mocker.patch("torch.utils.data.get_worker_info", return_value=MockWorker(id=5, num_workers=6))

    assert worker_info.is_parallel
    assert tuple(worker_info) == (5,)
    assert worker_info.id == 5
    assert worker_info.num_workers == 6


def test_unset_worker():
    assert not worker_info.is_parallel
    assert tuple(worker_info) == (0,)
    assert worker_info.id == 0
    assert worker_info.num_workers == 1
