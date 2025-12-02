from typing import Any

import pyarrow as pa
import pytest
import torch

from replay.data.utils.typing.dtype import torch_to_pyarrow
from replay.data.parquet.impl.flat_column import to_flat_columns
from replay.data.parquet.impl.named_columns import NamedColumns
from replay.data.parquet.impl.sequence_column import to_sequence_columns

TORCH_DTYPE_LIST: list[torch.dtype] = [torch.int8, torch.int64, torch.float32]


@pytest.mark.parametrize("a_type", TORCH_DTYPE_LIST)
@pytest.mark.parametrize("b_type", TORCH_DTYPE_LIST)
@pytest.mark.parametrize("c_type", TORCH_DTYPE_LIST)
def test_named_columns(a_type: torch.dtype, b_type: torch.dtype, c_type: torch.dtype):
    a_pa_type: pa.DataType = torch_to_pyarrow(a_type)
    b_pa_type: pa.DataType = torch_to_pyarrow(b_type)
    c_pa_type: pa.DataType = torch_to_pyarrow(c_type)
    cols: dict[str, pa.Array] = {
        "a": pa.array([0, 2, None, 1], type=a_pa_type),
        "b": pa.array([6, 4, -1, 3], type=b_pa_type),
    }
    seqs: dict[str, torch.Tensor] = {"c": pa.array([[0, 1], [], [2], [3, 4, 5]], type=pa.list_(c_pa_type))}

    metadata: dict[str, dict[str, Any]] = {
        "a": {"sequential": False},
        "b": {"sequential": False, "padding": -2},
        "c": {"sequential": True, "sequence_length": 2},
    }

    table: pa.Table = pa.Table.from_pydict({**cols, **seqs})

    named: NamedColumns = NamedColumns(columns={**to_flat_columns(table, metadata), **to_sequence_columns(table, metadata)})
    indices: torch.LongTensor = torch.asarray([2, 1, 0, 3, 2], dtype=torch.int64)
    result: dict[str, torch.Tensor] = named[indices]

    true_a_mask: torch.BoolTensor = torch.asarray([False, True, True, True, False], dtype=torch.bool)
    assert result["a_mask"].dtype == torch.bool
    assert torch.all(true_a_mask == result["a_mask"]).cpu().item()

    true_a_output: torch.Tensor = torch.asarray([-1, 2, 0, 1, -1], dtype=a_type)
    assert result["a"].dtype == a_type
    assert torch.all(true_a_output == result["a"]).cpu().item()

    assert result["b_mask"].dtype == torch.bool
    assert torch.all(result["b_mask"]).cpu().item()

    true_b_output: torch.Tensor = torch.asarray([-1, 4, 6, 3, -1], dtype=b_type)
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
