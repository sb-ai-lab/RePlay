import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq
import pytest
import torch
from hypothesis import (
    assume,
    example,
    given,
    settings,
    strategies as st,
)

from replay.data.nn.parquet.parquet_dataset import ParquetDataset
from replay.data.nn.parquet.utils.compute_length import (
    compute_fixed_size_batches_length,
    compute_fixed_size_generic_length,
)

settings.load_profile("fast")


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


@given(
    seed=st.integers(min_value=0, max_value=torch.iinfo(torch.int64).max),
    num_replicas=st.integers(min_value=1, max_value=4),
    batch_size=st.integers(min_value=1, max_value=32),
    partition_size=st.integers(min_value=16, max_value=64),
    partition_count=st.integers(min_value=1, max_value=32),
)
@example(67, 1, 8, 16, 8)
def test_parquet_dataset_length(
    seed: int, num_replicas: int, batch_size: int, partition_size: int, partition_count: int
) -> None:
    assume(partition_size >= batch_size)
    assume(partition_size % batch_size == 0)

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

        length = last_idx + 1
        assert length == len(dataset)

        generic_length = compute_fixed_size_generic_length(
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


@given(
    seed=st.integers(min_value=0, max_value=torch.iinfo(torch.int64).max),
    batch_size=st.integers(min_value=1, max_value=32),
    partition_size=st.integers(min_value=1, max_value=128),
)
@example(42, 4, 2)
@example(42, 4, 3)
def test_suboptimal_params(seed: int, batch_size: int, partition_size: int) -> None:
    assume(partition_size < batch_size or partition_size % batch_size != 0)

    with fragmented_parquet(seed, 1) as parquet_dir, pytest.warns(UserWarning) as record:
        _ = ParquetDataset(
            source=parquet_dir,
            metadata={"test_data": {}},
            partition_size=partition_size,
            batch_size=batch_size,
            replicas_info=FakeReplicasInfo(num_replicas=1),
        )

    assert any("Suboptimal parameters:" in warning.message.args[0] for warning in record)


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
    ],
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
            filesystem=path,
        )

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
            replicas_info=FakeReplicasInfo(num_replicas=1),
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
            replicas_info=FakeReplicasInfo(num_replicas=1),
        )

        with pytest.warns(UserWarning):
            dataset.iterator = list(dataset.iterator)
            dataset.compute_length()
