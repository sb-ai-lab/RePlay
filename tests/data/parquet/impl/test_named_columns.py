import pytest
import torch

from replay.data.parquet.impl.flat_column import FlatColumn
from replay.data.parquet.impl.named_columns import NamedColumns
from replay.data.parquet.impl.sequence_column import SequenceColumn

TORCH_DTYPE_LIST: list[torch.dtype] = [torch.int8, torch.int64, torch.float32]


@pytest.mark.parametrize("a_type", TORCH_DTYPE_LIST)
@pytest.mark.parametrize("b_type", TORCH_DTYPE_LIST)
@pytest.mark.parametrize("c_type", TORCH_DTYPE_LIST)
def test_named_columns(a_type: torch.dtype, b_type: torch.dtype, c_type: torch.dtype):
    cols_raw: dict[str, torch.Tensor] = {
        "a": torch.asarray([0, 2, -1, 1], dtype=a_type),
        "b": torch.asarray([6, 4, -1, 3], dtype=b_type),
        "a_mask": torch.asarray([1, 1, 0, 1], dtype=torch.bool),
    }
    cols: dict[str, FlatColumn] = {
        "a": FlatColumn(data=cols_raw["a"], mask=cols_raw["a_mask"]),
        "b": FlatColumn(data=cols_raw["b"]),
    }
    seqs_raw: dict[str, torch.Tensor] = {
        "c": torch.asarray([0, 1, 2, 3, 4, 5], dtype=c_type),
        "c_len": torch.asarray([2, 0, 1, 3], dtype=torch.int64),
    }
    seqs: dict[str, SequenceColumn] = {
        "c": SequenceColumn(data=seqs_raw["c"], lengths=seqs_raw["c_len"], sequence_length=2),
    }

    named: NamedColumns = NamedColumns(columns={**cols, **seqs})
    indices: torch.LongTensor = torch.asarray([2, 1, 0, 3, 2], dtype=torch.int64)
    result: dict[str, torch.Tensor] = named[indices]

    true_a_mask: torch.BoolTensor = cols_raw["a_mask"][indices]
    assert result["a_mask"].dtype == torch.bool
    assert torch.all(true_a_mask == result["a_mask"]).cpu().item()

    true_a_output: torch.BoolTensor = cols_raw["a"][indices]
    assert result["a"].dtype == a_type
    assert torch.all(true_a_output == result["a"]).cpu().item()

    assert result["b_mask"].dtype == torch.bool
    assert torch.all(result["b_mask"]).cpu().item()

    true_b_output: torch.BoolTensor = cols_raw["b"][indices]
    assert result["b"].dtype == b_type
    assert torch.all(true_b_output == result["b"]).cpu().item()

    true_c_mask: torch.BoolTensor = torch.asarray(
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

    true_c_output: torch.Tensor = torch.asarray(
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
