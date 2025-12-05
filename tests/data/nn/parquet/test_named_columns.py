from typing import Dict

import pytest
import torch

from replay.data.nn.parquet.impl.array_1d_column import Array1DColumn
from replay.data.nn.parquet.impl.flat_column import FlatColumn
from replay.data.nn.parquet.impl.named_columns import NamedColumns
from replay.data.utils.batching import UniformBatching
from tests.data.nn.parquet.conftest import BatchGenerator

Batch = Dict[str, torch.Tensor]


@pytest.mark.parametrize("seed", [1, 3, 7, 19])
@pytest.mark.parametrize("batch_size", [1, 1024, 4096])
def test_sincos_one_batch(seed: int, batch_size: int) -> None:
    batch_generator: BatchGenerator = BatchGenerator(
        generator=torch.Generator().manual_seed(seed),
        batch_size=batch_size,
    )

    gtr_generator: BatchGenerator = BatchGenerator(
        generator=torch.Generator().manual_seed(seed),
        batch_size=batch_size,
    )

    batch: Batch = batch_generator.generate_batch()
    sin_col: Array1DColumn = Array1DColumn(
        data=batch["sin_sin"],
        lengths=batch["sin_length"],
        shape=batch_generator.max_length,
        padding=-2,
    )
    phase_col: FlatColumn = FlatColumn(data=batch["phase"])
    frequency_col: FlatColumn = FlatColumn(data=batch["frequency"])
    named: NamedColumns = NamedColumns(
        {
            "sin_sin": sin_col,
            "phase": phase_col,
            "frequency": frequency_col,
        },
    )
    ids: torch.LongTensor = torch.arange(0, batch_size, dtype=torch.int64)

    batch: Batch = named[ids]
    gtr_batch: Batch = gtr_generator.generate_padded()

    assert torch.allclose(batch["sin_sin"], gtr_batch["sin_sin"])
    assert torch.allclose(batch["phase"][:, None], gtr_batch["phase"])
    assert torch.allclose(batch["frequency"][:, None], gtr_batch["frequency"])


@pytest.mark.parametrize("seed", [1, 7, 19])
@pytest.mark.parametrize("batch_size", [256, 512])
@pytest.mark.parametrize("sub_batch_size", [1, 8, 256])
def test_sincos_seq_batch(seed: int, batch_size: int, sub_batch_size: int):
    assert batch_size % sub_batch_size == 0

    batch_generator: BatchGenerator = BatchGenerator(
        generator=torch.Generator().manual_seed(seed),
        batch_size=batch_size,
    )

    gtr_generator: BatchGenerator = BatchGenerator(
        generator=torch.Generator().manual_seed(seed),
        batch_size=batch_size,
    )

    batch: Batch = batch_generator.generate_batch()
    sin_col: Array1DColumn = Array1DColumn(
        data=batch["sin_sin"],
        lengths=batch["sin_length"],
        shape=batch_generator.max_length,
        padding=-2,
    )
    phase_col: FlatColumn = FlatColumn(data=batch["phase"])
    frequency_col: FlatColumn = FlatColumn(data=batch["frequency"])
    named: NamedColumns = NamedColumns(
        {
            "sin_sin": sin_col,
            "phase": phase_col,
            "frequency": frequency_col,
        },
    )

    gtr_batch: Batch = gtr_generator.generate_padded()
    batching: UniformBatching = UniformBatching(batch_size, sub_batch_size)

    first: int
    last: int
    for first, last in iter(batching):
        ids: torch.LongTensor = torch.arange(first, last, dtype=torch.int64)

        sub_batch: Batch = named[ids]
        gtr_sub_batch: Batch = {k: v[first:last, ...] for k, v in gtr_batch.items()}

        assert torch.allclose(sub_batch["sin_sin"], gtr_sub_batch["sin_sin"])
        assert torch.allclose(sub_batch["phase"][:, None], gtr_sub_batch["phase"])
        assert torch.allclose(sub_batch["frequency"][:, None], gtr_sub_batch["frequency"])
