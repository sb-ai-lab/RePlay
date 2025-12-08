from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import torch

from replay.constants.device import DEFAULT_DEVICE
from replay.constants.metadata import DEFAULT_PADDING
from replay.data.nn.parquet.metadata import (
    Metadata,
    get_2d_array_columns,
    get_padding,
    get_shape,
)
from replay.data.utils.typing.dtype import pyarrow_to_torch

from .column_protocol import OutputType
from .indexing import get_mask, get_offsets
from .utils import ensure_mutable


class Array2DColumn:
    """
    A representation of a 2D array column, containing nested
    lists of numbers of varying length in each of its rows.
    """

    def __init__(
        self,
        data: torch.Tensor,
        outer_lengths: torch.LongTensor,
        inner_lengths: torch.LongTensor,
        shape: list[int],
        padding: Any = DEFAULT_PADDING,
    ) -> None:
        """
        :param data: A tensor containing column data.
        :param outer_lengths: A tensor containing inner lengths (dim 0) of each individual row array.
        :param inner_lengths: A tensor containing lengths (dim 1) of each individual row array.
        :param shape: An integer or list of integers representing the target array shapes.
        :param padding: Padding value to use to fill null values and match target shape.
            Default: value of `DEFAULT_PADDING`

        :raises ValueError: If the shape provided is not two-dimensional.
        """
        self.padding = padding
        self.data = data
        self.inner_offsets = get_offsets(inner_lengths)
        self.outer_offsets = get_offsets(outer_lengths)
        if len(shape) != 2:
            msg = f"Array2DColumn accepts a shape of size (2,) only. Got {shape=}"
            raise ValueError(msg)
        self.shape: list[int] = shape

    @property
    def length(self) -> int:
        return torch.numel(self.outer_offsets) - 1

    def __len__(self) -> int:
        return self.length

    @property
    def device(self) -> torch.device:
        assert self.data.device == self.inner_offsets.device
        assert self.data.device == self.outer_offsets.device
        return self.inner_offsets.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def __getitem__(self, indices: torch.LongTensor) -> OutputType:
        indices = indices.to(device=self.device)
        outer_mask, outer_output = get_mask(indices, self.outer_offsets, self.shape[0])
        left_bound = outer_output.min().item()
        right_bound = outer_output.max().item()
        outer_output -= left_bound

        inner_indices = torch.arange(left_bound, right_bound + 1, device=indices.device)
        inner_mask, output = get_mask(inner_indices, self.inner_offsets, self.shape[1])

        final_indices = output[outer_output]
        inner_final_mask = inner_mask[outer_output]

        unmasked_values = torch.take(self.data, final_indices)
        outer_final_mask = outer_mask.unsqueeze(-1).repeat(1, 1, unmasked_values.size(-1))
        mask = inner_final_mask * outer_final_mask

        masked_values = torch.where(mask, unmasked_values, self.padding)
        assert masked_values.device == self.device
        assert masked_values.dtype == self.dtype
        return (mask, masked_values)


def to_torch(
    array: pa.Array,
    device: torch.device = DEFAULT_DEVICE,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
    """
    Converts a PyArrow array into a PyTorch tensor.

    :param array: Original PyArow array.
    :param device: Target device to send the resulting tensor to. Default: value of `DEFAULT_DEVICE`.

    :return: A PyTorch tensor obtained from original array.
    """
    flatten_dim0 = pc.list_flatten(array)
    flatten = pc.list_flatten(flatten_dim0)

    outer_lengths = pc.list_value_length(array).cast(pa.int64())
    inner_lengths = pc.list_value_length(flatten_dim0).cast(pa.int64())

    # Copying to be mutable
    flatten_torch = torch.asarray(
        ensure_mutable(flatten.to_numpy()),
        device=device,
        dtype=pyarrow_to_torch(flatten.type),
    )

    # Copying to be mutable
    outer_lengths_torch = torch.asarray(
        ensure_mutable(outer_lengths.to_numpy()),
        device=device,
        dtype=torch.int64,
    )
    inner_lengths_torch = torch.asarray(
        ensure_mutable(inner_lengths.to_numpy()),
        device=device,
        dtype=torch.int64,
    )
    return (outer_lengths_torch, inner_lengths_torch, flatten_torch)


def to_array_2d_columns(
    data: pa.RecordBatch,
    metadata: Metadata,
    device: torch.device = DEFAULT_DEVICE,
) -> dict[str, Array2DColumn]:
    """
    Converts a PyArrow batch of data to a set of `Array2DColums`s.
    This function filters only those columns matching its format from the full batch.

    :param data: A PayArrow batch of column data.
    :param metadata: Metadata containing information about columns' formats.
    :param device: Target device to send column tensors to. Default: value of `DEFAULT_DEVICE`

    :return: A dict of tensors containing dataset's numeric columns.
    """
    result = {}

    for column_name in get_2d_array_columns(metadata):
        outer_lengths, inner_lengths, torch_array = to_torch(data.column(column_name), device=device)
        result[column_name] = Array2DColumn(
            data=torch_array,
            outer_lengths=outer_lengths,
            inner_lengths=inner_lengths,
            padding=get_padding(metadata, column_name),
            shape=get_shape(metadata, column_name),
        )
    return result
