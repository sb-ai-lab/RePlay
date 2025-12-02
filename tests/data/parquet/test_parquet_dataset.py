from contextlib import contextmanager
import re
import warnings
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as fs
import pytest
import torch

from replay.data.parquet.parquet_dataset import ParquetDataset
from replay.data.parquet.utils.compute_length import (
    compute_fixed_size_batches_length,
    compute_fixed_size_generic_length,
)


@dataclass
class FakeReplicasInfo:
    num_replicas: int = 1
    curr_replica: int = 0

@contextmanager
def fragmented_parquet(seed: int, partition_count: int):
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    fragment_sizes: list[int] = torch.randint(low=1, high=31, size=(partition_count,), generator=generator).tolist()

    tmpdir = TemporaryDirectory()
    for i, fragment_size in enumerate(fragment_sizes):
        test_data: torch.Tensor = torch.arange(fragment_size)
        table: pa.Table = pa.table({"test_data": test_data.tolist()})
        pq.write_table(table, f"{tmpdir.name}/fragment_{i}.parquet")
    try:
        yield tmpdir.name
    finally:
        tmpdir.cleanup()


@pytest.mark.parametrize("seed", [1, 42, 777])
@pytest.mark.parametrize("num_replicas", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [1, 2, 7, 19])
@pytest.mark.parametrize("partition_size", [1, 3, 5, 17])
@pytest.mark.parametrize("partition_count", [1, 2, 7, 19])
def test_parquet_dataset_length(
    seed: int, num_replicas: int, batch_size: int, partition_size: int, partition_count: int
) -> None:
    with fragmented_parquet(seed, partition_count) as parquet_dir:
        fake_replicas_info: FakeReplicasInfo = FakeReplicasInfo(num_replicas=num_replicas)
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message=re.escape("Suboptimal parameters:") + ".*",
        )
        dataset: ParquetDataset = ParquetDataset(
            source=parquet_dir,
            metadata={"test_data": {}},
            partition_size=partition_size,
            batch_size=batch_size,
            replicas_info=fake_replicas_info,
        )

        last_idx = 0
        for i, _ in enumerate(dataset):
            last_idx = i

        length: int = last_idx + 1
        assert length == len(dataset)

        generic_length: int = compute_fixed_size_generic_length(
            iterable=dataset.iterator,
            num_replicas=num_replicas,
            batch_size=batch_size,
        )

        assert length == generic_length

        batches_length: int = compute_fixed_size_batches_length(
            iterable=dataset.iterator,
            num_replicas=num_replicas,
            batch_size=batch_size,
        )

        assert length == batches_length

def test_non_computable_length():
    with fragmented_parquet(42, 1) as parquet_dir:
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message=re.escape("Suboptimal parameters:") + ".*",
        )
        dataset = ParquetDataset(
            source=parquet_dir,
            metadata={"test_data": {}},
            partition_size=10,
            batch_size=5,
            replicas_info=FakeReplicasInfo(num_replicas=1),
        )

        with pytest.raises(TypeError):
            dataset.do_compute_length = False
            len(dataset)


@pytest.mark.parametrize(
    "path, expected_fs",
    [
        ("file:///hey/foo.parquet", fs.LocalFileSystem),
        ("mock://foo.parquet", fs._MockFileSystem),
    ]
)
def test_init_with_fs(path, expected_fs, mocker):
    mocker.patch("pyarrow.dataset.dataset", return_value=None)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message=re.escape("Suboptimal parameters:") + ".*",
        )
        dataset = ParquetDataset(
            source=path,
            metadata={"test_data": {}},
            partition_size=10,
            batch_size=5,
            replicas_info=FakeReplicasInfo(num_replicas=1),
            filesystem=path
        )

        print(type(dataset.filesystem))
        assert isinstance(dataset.filesystem, expected_fs)


def test_replica_invalidation(mocker):
    with fragmented_parquet(42, 1) as parquet_dir:
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message=re.escape("Suboptimal parameters:") + ".*",
        )
        dataset = ParquetDataset(
            source=parquet_dir,
            metadata={"test_data": {}},
            partition_size=10,
            batch_size=5,
            replicas_info=FakeReplicasInfo(num_replicas=1)
        )

        dataset.cached_lengths = {1: [3, 5]}
        assert dataset.compute_length() == [3, 5]
        with pytest.warns(UserWarning):
            dataset.replicas_info = FakeReplicasInfo(num_replicas=2)
            dataset.compute_length()


def test_generic_iterator_size_length():
    with fragmented_parquet(42, 1) as parquet_dir:
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message=re.escape("Suboptimal parameters:") + ".*",
        )
        dataset = ParquetDataset(
            source=parquet_dir,
            metadata={"test_data": {}},
            partition_size=10,
            batch_size=5,
            replicas_info=FakeReplicasInfo(num_replicas=1)
        )

        with pytest.warns(UserWarning):
            dataset.iterator = list(dataset.iterator)
            dataset.compute_length()