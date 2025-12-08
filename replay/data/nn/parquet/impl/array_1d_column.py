from typing import Any, Union

import pyarrow as pa
import pyarrow.compute as pc
import torch

from replay.constants.device import DEFAULT_DEVICE
from replay.constants.metadata import DEFAULT_PADDING
from replay.data.nn.parquet.metadata import (
    Metadata,
    get_1d_array_columns,
    get_padding,
    get_shape,
)
from replay.data.utils.typing.dtype import pyarrow_to_torch

from .column_protocol import OutputType
from .indexing import get_mask, get_offsets
from .utils import ensure_mutable


class Array1DColumn:
    """
    A representation of a 1D array column, containing a
    list of numbers of varying length in each of its rows.
    """

    def __init__(
        self,
        data: torch.Tensor,
        lengths: torch.LongTensor,
        shape: Union[int, list[int]],
        padding: Any = DEFAULT_PADDING,
    ) -> None:
        """
        :param data: A tensor containing column data.
        :param lengths: A tensor containing lengths of each individual row array.
        :param shape: An integer or list of integers representing the target array shapes.
        :param padding: Padding value to use to fill null values and match target shape.
            Default: value of `DEFAULT_PADDING`

        :raises ValueError: If the shape provided is not one-dimensional.
        """
        if isinstance(shape, list) and len(shape) > 1:
            msg = f"Array1DColumn accepts a shape of size (1,) only. Got {shape=}"
            raise ValueError(msg)

        self.padding = padding
        self.data = data
        self.offsets = get_offsets(lengths)
        self.shape = shape[0] if isinstance(shape, list) else shape
        assert self.length == torch.numel(lengths)

    @property
    def length(self) -> int:
        return torch.numel(self.offsets) - 1

    def __len__(self) -> int:
        return self.length

    @property
    def device(self) -> torch.device:
        assert self.data.device == self.offsets.device
        return self.offsets.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def __getitem__(self, indices: torch.LongTensor) -> OutputType:
        indices = indices.to(device=self.device)
        mask, output = get_mask(indices, self.offsets, self.shape)

        # TODO: Test this for both 1d and 2d arrays. Add same check in 2d arrays
        if self.data.numel() == 0:
            mask = torch.zeros((indices.size(0), self.shape), dtype=torch.bool, device=self.device)
            output = torch.ones((indices.size(0), self.shape), dtype=torch.bool, device=self.device) * self.padding
            return mask, output

        unmasked_values = torch.take(self.data, output)
        masked_values = torch.where(mask, unmasked_values, self.padding)
        assert masked_values.device == self.device
        assert masked_values.dtype == self.dtype
        return (mask, masked_values)


def to_torch(array: pa.Array, device: torch.device = DEFAULT_DEVICE) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a PyArrow array into a PyTorch tensor.

    :param array: Original PyArow array.
    :param device: Target device to send the resulting tensor to. Default: value of `DEFAULT_DEVICE`.

    :return: A PyTorch tensor obtained from original array.
    """
    flatten = pc.list_flatten(array)
    lengths = pc.list_value_length(array).cast(pa.int64())

    # Copying to be mutable
    flatten_torch = torch.asarray(
        ensure_mutable(flatten.to_numpy()),
        device=device,
        dtype=pyarrow_to_torch(flatten.type),
    )

    # Copying to be mutable
    lengths_torch = torch.asarray(
        ensure_mutable(lengths.to_numpy()),
        device=device,
        dtype=torch.int64,
    )
    return (lengths_torch, flatten_torch)


def to_array_1d_columns(
    data: pa.RecordBatch,
    metadata: Metadata,
    device: torch.device = DEFAULT_DEVICE,
) -> dict[str, Array1DColumn]:
    """
    Converts a PyArrow batch of data to a set of `Array1DColums`s.
    This function filters only those columns matching its format from the full batch.

    :param data: A PayArrow batch of column data.
    :param metadata: Metadata containing information about columns' formats.
    :param device: Target device to send column tensors to. Default: value of `DEFAULT_DEVICE`

    :return: A dict of tensors containing dataset's numeric columns.
    """
    result: dict[str, Array1DColumn] = {}

    for column_name in get_1d_array_columns(metadata):
        lengths, torch_array = to_torch(data.column(column_name), device=device)
        result[column_name] = Array1DColumn(
            data=torch_array,
            lengths=lengths,
            padding=get_padding(metadata, column_name),
            shape=get_shape(metadata, column_name),
        )
    return result
