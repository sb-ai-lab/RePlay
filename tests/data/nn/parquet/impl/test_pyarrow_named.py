import pyarrow as pa
import torch
from hypothesis import (
    given,
    settings,
    strategies as st,
)

from replay.data.nn.parquet.impl.array_1d_column import to_array_1d_columns
from replay.data.nn.parquet.impl.array_2d_column import to_array_2d_columns
from replay.data.nn.parquet.impl.numeric_column import to_numeric_columns
from replay.data.nn.parquet.impl.named_columns import NamedColumns
from replay.data.utils.typing.dtype import torch_to_pyarrow

settings.load_profile("fast")


TORCH_DTYPE_LIST = [torch.int8, torch.int64, torch.float32, torch.float32]


@given(
    a_type=st.sampled_from(TORCH_DTYPE_LIST),
    b_type=st.sampled_from(TORCH_DTYPE_LIST),
    c_type=st.sampled_from(TORCH_DTYPE_LIST),
    d_type=st.sampled_from(TORCH_DTYPE_LIST),
)
def test_named_columns(a_type: torch.dtype, b_type: torch.dtype, c_type: torch.dtype, d_type: torch.dtype):
    a_pa_type = torch_to_pyarrow(a_type)
    b_pa_type = torch_to_pyarrow(b_type)
    c_pa_type = torch_to_pyarrow(c_type)
    d_pa_type = torch_to_pyarrow(d_type)

    cols = {
        "a": pa.array([0, 2, None, 1], type=a_pa_type),
        "b": pa.array([6, 4, -1, 3], type=b_pa_type),
    }
    seqs_1d = {"c": pa.array([[0, 1], [], [2], [3, 4, 5]], type=pa.list_(c_pa_type))}
    seqs_2d = {
        "d": pa.array([[[0, 1]], [[], []], [[2], [3], [4]], [[3, 4, 5], [1, 4]]], type=pa.list_(pa.list_(d_pa_type)))
    }

    metadata = {
        "a": {},
        "b": {"padding": -2},
        "c": {"shape": 2},
        "d": {"shape": [2, 3]},
    }

    table = pa.Table.from_pydict({**cols, **seqs_1d, **seqs_2d})

    named = NamedColumns(
        columns={
            **to_numeric_columns(table, metadata),
            **to_array_1d_columns(table, metadata),
            **to_array_2d_columns(table, metadata),
        }
    )
    indices = torch.asarray([2, 1, 0, 3, 2], dtype=torch.int64)
    result = named[indices]

    assert all(len(column) == 4 for column in named.columns.values())

    true_a_mask = torch.asarray([False, True, True, True, False], dtype=torch.bool)
    assert result["a_mask"].dtype == torch.bool
    assert torch.all(true_a_mask == result["a_mask"]).cpu().item()

    true_a_output = torch.asarray([-1, 2, 0, 1, -1], dtype=a_type)
    assert result["a"].dtype == a_type
    assert torch.all(true_a_output == result["a"]).cpu().item()

    assert result["b_mask"].dtype == torch.bool
    assert torch.all(result["b_mask"]).cpu().item()

    true_b_output = torch.asarray([-1, 4, 6, 3, -1], dtype=b_type)
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

    true_d_mask = torch.asarray(
        [
            [[False, False, True], [False, False, True]],
            [[False, False, False], [False, False, False]],
            [[False, False, False], [False, True, True]],
            [[True, True, True], [False, True, True]],
            [[False, False, True], [False, False, True]],
        ],
        dtype=torch.bool,
    )
    assert result["d_mask"].dtype == torch.bool
    assert torch.all(true_d_mask == result["d_mask"]).cpu().item()

    true_d_output = torch.asarray(
        [
            [[-1, -1, 3], [-1, -1, 4]],
            [[-1, -1, -1], [-1, -1, -1]],
            [[-1, -1, -1], [-1, 0, 1]],
            [[3, 4, 5], [-1, 1, 4]],
            [[-1, -1, 3], [-1, -1, 4]],
        ],
        dtype=d_type,
    )
    assert result["d"].dtype == d_type
    assert torch.all(true_d_output == result["d"]).cpu().item()
