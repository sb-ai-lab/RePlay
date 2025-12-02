import os
import tempfile
from typing import Any, Dict

import pyarrow.dataset as ds
import pytest
import torch

from replay.data.parquet.iterable_dataset import IterableDataset
from replay.data.parquet.partitioned_iterable_dataset import PartitionedIterableDataset as TorchPartitionedIterableDataset
from replay.data.parquet.iterator import BatchesIterator
from tests.data.parquet.conftest import BatchGenerator, make_metadata, write_dataset


@pytest.mark.parametrize("seed", [42, 777])
@pytest.mark.parametrize("batch_size", [5, 7, 9, 12])
@pytest.mark.parametrize("batch_count", [3, 4, 8, 25])
def test_dataset_pipeline(seed: int, batch_size: int, batch_count: int):
    gtr_generator: BatchGenerator = BatchGenerator(
        generator=torch.Generator().manual_seed(seed),
        batch_size=batch_size,
    )
    batch_generator: BatchGenerator = BatchGenerator(
        generator=torch.Generator().manual_seed(seed),
        batch_size=batch_size,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        path: str = os.path.join(temp_dir, "partitioned")
        os.makedirs(path)
        write_dataset(batch_generator, path, batch_count)
        dataset: ds.Dataset = ds.dataset(path, format="parquet")

        metadata: Any = make_metadata(batch_generator)

        batches_iterator: BatchesIterator = BatchesIterator(
            batch_size=batch_size,
            metadata=metadata,
            dataset=dataset,
        )

        partitioned_dataset: TorchPartitionedIterableDataset = TorchPartitionedIterableDataset(
            iterable=batches_iterator,
            batch_size=batch_size,
        )

        batch: Dict[str, torch.Tensor]
        for batch in partitioned_dataset:
            gtr_batch: Dict[str, torch.Tensor] = gtr_generator.generate_padded()

            key: str
            for key in metadata:
                shape: Any = gtr_batch[key].shape
                element: torch.Any = batch[key].reshape(shape)
                assert torch.allclose(element, gtr_batch[key])


def test_internal_iterator():
    BATCH_SIZE = 8
    BATCH_COUNT = 4
    SEED = 42

    batch_generator = BatchGenerator(
        generator=torch.Generator().manual_seed(SEED),
        batch_size=BATCH_SIZE,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "partitioned")
        os.makedirs(path)
        write_dataset(batch_generator, path, BATCH_COUNT)
        dataset = ds.dataset(path, format="parquet")

        metadata = make_metadata(batch_generator)

        batches_iterator = BatchesIterator(
            batch_size=BATCH_SIZE,
            metadata=metadata,
            dataset=dataset,
        )

        partitioned_dataset = TorchPartitionedIterableDataset(
            iterable=batches_iterator,
            batch_size=BATCH_SIZE,
        )

        inner_iterable = IterableDataset(
            named_columns=next(iter(partitioned_dataset.iterable)),
            generator=partitioned_dataset.generator,
            batch_size=partitioned_dataset.batch_size,
            replicas_info=partitioned_dataset.replicas_info,
        )

        assert inner_iterable.device == torch.empty(1).device
        assert len(inner_iterable) == 1

