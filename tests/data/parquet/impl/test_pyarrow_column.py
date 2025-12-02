import pyarrow as pa
import pytest
import torch

from replay.data.utils.typing.dtype import torch_to_pyarrow
from replay.data.parquet.impl.flat_column import FlatColumn, to_torch

TORCH_DTYPE_LIST: list[torch.dtype] = [
    torch.int8,
    torch.int16,
    torch.int64,
    torch.float32,
    torch.float64,
]


@pytest.mark.parametrize("seed", [1, 42, 333])
@pytest.mark.parametrize("elem_count", [1, 23, 127])
@pytest.mark.parametrize("torch_dtype", TORCH_DTYPE_LIST)
def test_column(seed: int, elem_count: int, torch_dtype: torch.dtype):
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    data: torch.Tensor = torch.randint(low=-10, high=+10, size=(elem_count,), generator=gen, dtype=torch_dtype)
    array: pa.Array = pa.array(data.cpu().tolist(), type=torch_to_pyarrow(torch_dtype))
    mask, torch_array = to_torch(array)
    column: FlatColumn = FlatColumn(data=torch_array, mask=mask)

    indices: torch.LongTensor = torch.randint(
        low=0,
        high=elem_count,
        size=(int(1.5 * elem_count),),
        generator=gen,
        dtype=torch.int64,
    )

    output: torch.Tensor
    _, output = column[indices]
    true_output: torch.Tensor = data[indices]

    assert torch.all(output == true_output).cpu().item()


@pytest.mark.parametrize("seed", [1, 42, 333])
@pytest.mark.parametrize("elem_count", [1, 23, 127])
@pytest.mark.parametrize("torch_dtype", TORCH_DTYPE_LIST)
def test_column_with_nulls(seed: int, elem_count: int, torch_dtype: torch.dtype):
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    data: torch.Tensor = torch.randint(low=-10, high=+10, size=(elem_count,), generator=gen, dtype=torch_dtype)
    not_nulls: torch.BoolTensor = torch.randint(low=-10, high=+10, size=(elem_count,), generator=gen, dtype=torch_dtype) < 0
    iterable = zip(data.cpu().tolist(), not_nulls.cpu().tolist(), strict=False)
    data_list = [value if not_null else None for value, not_null in iterable]
    array: pa.Array = pa.array(data_list, type=torch_to_pyarrow(torch_dtype))
    mask, torch_array = to_torch(array)
    column: FlatColumn = FlatColumn(data=torch_array, mask=mask)

    indices: torch.LongTensor = torch.randint(
        low=0,
        high=elem_count,
        size=(int(1.5 * elem_count),),
        generator=gen,
        dtype=torch.int64,
    )

    mask: torch.BoolTensor
    output: torch.Tensor
    mask, output = column[indices]
    true_mask: torch.BoolTensor = not_nulls[indices]
    true_output: torch.Tensor = torch.where(true_mask, data[indices], -1)

    assert torch.all(mask == true_mask).cpu().item()
    assert torch.all(output == true_output).cpu().item()
