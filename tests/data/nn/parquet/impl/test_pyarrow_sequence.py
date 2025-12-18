import pyarrow as pa
import pytest
import torch

from replay.data.nn.parquet.impl.array_1d_column import Array1DColumn, to_torch
from replay.data.utils.typing.dtype import torch_to_pyarrow

TORCH_DTYPE_LIST: list[torch.dtype] = [
    torch.int8,
    torch.int16,
    torch.int64,
    torch.float32,
    torch.float64,
]


@pytest.mark.parametrize("torch_dtype", TORCH_DTYPE_LIST)
def test_pyarrow_array1d(torch_dtype: torch.dtype):
    pa_dtype: pa.DataType = torch_to_pyarrow(torch_dtype)
    data: list = [[1, 3, 4], [6, 5], [], [9, 7], [3]]
    array: pa.Array = pa.array(data, type=pa.list_(pa_dtype))
    lengths, torch_array = to_torch(array)
    sequence_column = Array1DColumn(
        data=torch_array,
        lengths=lengths,
        shape=2,
    )

    ids: torch.LongTensor = torch.asarray([1, 2, 1, 0, 4], device=sequence_column.device, dtype=torch.int64)

    mask: torch.BoolTensor
    output: torch.Tensor
    mask, output = sequence_column[ids]

    mask_gtr: torch.BoolTensor = torch.asarray(
        [
            [True, True],
            [False, False],
            [True, True],
            [True, True],
            [False, True],
        ],
        device=sequence_column.device,
        dtype=torch.bool,
    )
    assert torch.all(mask_gtr == mask).cpu().item()

    output_gtr: torch.Tensor = torch.asarray(
        [
            [6, 5],
            [-1, -1],
            [6, 5],
            [3, 4],
            [-1, 3],
        ],
        device=sequence_column.device,
        dtype=torch_dtype,
    )

    assert torch.all(output_gtr == output).cpu().item()
