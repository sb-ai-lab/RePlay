import pytest
import torch

from replay.data.nn.parquet.impl.array_1d_column import Array1DColumn
from replay.data.nn.parquet.impl.flat_column import FlatColumn
from replay.data.nn.parquet.impl.named_columns import NamedColumns, deduce_device, deduce_length
from replay.data.utils.batching import UniformBatching
from tests.data.nn.parquet.conftest import BatchGenerator

TORCH_DTYPE_LIST: list[torch.dtype] = [torch.int8, torch.int64, torch.float32]
Batch = dict[str, torch.Tensor]


@pytest.mark.parametrize("a_type", TORCH_DTYPE_LIST)
@pytest.mark.parametrize("b_type", TORCH_DTYPE_LIST)
@pytest.mark.parametrize("c_type", TORCH_DTYPE_LIST)
def test_named_columns(a_type: torch.dtype, b_type: torch.dtype, c_type: torch.dtype):
    cols_raw = {
        "a": torch.asarray([0, 2, -1, 1], dtype=a_type),
        "b": torch.asarray([6, 4, -1, 3], dtype=b_type),
        "a_mask": torch.asarray([1, 1, 0, 1], dtype=torch.bool),
    }
    cols = {
        "a": FlatColumn(data=cols_raw["a"], mask=cols_raw["a_mask"]),
        "b": FlatColumn(data=cols_raw["b"]),
    }
    seqs_raw: dict[str, torch.Tensor] = {
        "c": torch.asarray([0, 1, 2, 3, 4, 5], dtype=c_type),
        "c_len": torch.asarray([2, 0, 1, 3], dtype=torch.int64),
    }
    seqs = {
        "c": Array1DColumn(data=seqs_raw["c"], lengths=seqs_raw["c_len"], shape=2),
    }

    named = NamedColumns(columns={**cols, **seqs})
    indices = torch.asarray([2, 1, 0, 3, 2], dtype=torch.int64)
    result = named[indices]

    true_a_mask = cols_raw["a_mask"][indices]
    assert result["a_mask"].dtype == torch.bool
    assert torch.all(true_a_mask == result["a_mask"]).cpu().item()

    true_a_output = cols_raw["a"][indices]
    assert result["a"].dtype == a_type
    assert torch.all(true_a_output == result["a"]).cpu().item()

    assert result["b_mask"].dtype == torch.bool
    assert torch.all(result["b_mask"]).cpu().item()

    true_b_output = cols_raw["b"][indices]
    assert result["b"].dtype == b_type
    assert torch.all(true_b_output == result["b"]).cpu().item()

    true_c_mask = torch.asarray(
        [
            [False, True],
            [False, False],
            [True, True],
            [True, True],
            [False, True],
        ],
        dtype=torch.bool,
    )
    assert result["c_mask"].dtype == torch.bool
    assert torch.all(true_c_mask == result["c_mask"]).cpu().item()

    true_c_output = torch.asarray(
        [
            [-1, 2],
            [-1, -1],
            [0, 1],
            [4, 5],
            [-1, 2],
        ],
        dtype=c_type,
    )
    assert result["c"].dtype == c_type
    assert torch.all(true_c_output == result["c"]).cpu().item()


def test_mismatching_lengths():
    with pytest.raises(RuntimeError):
        deduce_length(
            [
                FlatColumn(torch.asarray([0, 2, -1])),
                FlatColumn(torch.asarray([6, 4, -1, 3])),
            ]
        )


@pytest.mark.skip("Driver-dependent test. Needs a rework.")
def test_mismatching_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", True)

    with pytest.raises(RuntimeError) as err:
        deduce_device(
            [
                FlatColumn(torch.asarray([0, 2, -1, 1], device="cpu")),
                FlatColumn(torch.asarray([6, 4, -1, 3], device="cuda")),
            ]
        )
    assert "Columns must be all on the same device." in str(err.value)


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
