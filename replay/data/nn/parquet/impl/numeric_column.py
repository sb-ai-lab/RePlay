from typing import Any, Optional

import pyarrow as pa
import torch

from replay.constants.device import DEFAULT_DEVICE
from replay.constants.metadata import DEFAULT_PADDING
from replay.data.nn.parquet.metadata import Metadata, get_numeric_columns
from replay.data.utils.typing.dtype import pyarrow_to_torch

from .column_protocol import OutputType
from .utils import ensure_mutable


class NumericColumn:
    """A representation of a numeric column, containing a single number in each of its rows."""

    def __init__(
        self,
        data: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        padding: Any = DEFAULT_PADDING,
    ) -> None:
        """
        :param data: A tensor containing column data.
        :param mask: A mask tensor to differentiate real values from paddings. Default: `None`.
        :param padding: Padding to use for future indexing of non-existent data. Default: value of `DEFAULT_PADDING`.
        """
        self.padding: Any = padding
        self.data = data
        self.mask = mask

    @property
    def length(self) -> int:
        result = torch.numel(self.data)
        if self.mask is not None:
            assert result == torch.numel(self.mask)
        return result

    def __len__(self) -> int:
        return self.length

    @property
    def device(self) -> torch.device:
        result = self.data.device
        if self.mask is not None:
            assert result == self.mask.device
        return result

    def _get_mask(self, indices: torch.LongTensor) -> torch.BoolTensor:
        mask = torch.ones_like(indices, dtype=torch.bool) if self.mask is None else self.mask[indices]
        return mask

    def __getitem__(self, indices: torch.LongTensor) -> OutputType:
        indices = indices.to(device=self.device)
        mask = self._get_mask(indices)
        output = torch.where(mask, self.data[indices], self.padding)
        return (mask, output)


def to_torch(array: pa.Array, device: torch.device = DEFAULT_DEVICE, padding: Any = DEFAULT_PADDING) -> OutputType:
    """
    Converts a PyArrow array into a PyTorch tensor.

    :param array: Original PyArow array.
    :param device: Target device to send the resulting tensor to. Default: value of `DEFAULT_DEVICE`.
    :param padding: Value to fill null values with. Default: value of to `DEFAULT_PADDING`.

    :return: A PyTorch tensor obtained from original array.
    """
    dtype = pyarrow_to_torch(array.type)

    mask_torch = None
    if array.null_count > 0:
        mask_torch = torch.asarray(
            ensure_mutable(array.is_valid().to_numpy(zero_copy_only=False)),
            device=device,
            dtype=torch.bool,
        )

    array_torch = torch.asarray(
        ensure_mutable(array.fill_null(padding).to_numpy()),
        device=device,
        dtype=dtype,
    )
    return (mask_torch, array_torch)


def to_numeric_columns(
    data: pa.RecordBatch,
    metadata: Metadata,
    device: torch.device = DEFAULT_DEVICE,
    padding: Any = DEFAULT_PADDING,
) -> dict[str, NumericColumn]:
    """
    Converts a PyArrow batch of data to a set of `NumericColumn`s.
    This function filters only those columns matching its format from the full batch.

    :param data: A PayArrow batch of column data.
    :param metadata: Metadata containing information about columns' formats.
    :param device: Target device to send column tensors to. Default: value of `DEFAULT_DEVICE`
    :param padding: Padding to use for future indexing of non-existent data. Default: value of `DEFAULT_PADDING`.

    :return: A dict of tensors containing dataset's numeric columns.
    """
    result = {}
    for column_name in get_numeric_columns(metadata):
        mask, torch_array = to_torch(data.column(column_name), device, padding)
        result[column_name] = NumericColumn(data=torch_array, mask=mask, padding=padding)
    return result
