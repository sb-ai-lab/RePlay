from collections.abc import Sequence
from typing import Callable

import torch

from replay.data.nn.parquet.impl.masking import DEFAULT_MAKE_MASK_NAME

from .column_protocol import ColumnProtocol

Batch = dict[str, torch.Tensor]


def deduce_device(columns: Sequence[ColumnProtocol]) -> torch.device:
    """
    Sanity check for matching devices on all of dataset's columns.

    :param columns: A list of dataset's column data.
    :raises RuntimeError: If any of the columns have mismatching devices.
    :return: The determined columns' device.
    """
    assert len(columns) > 0
    device = columns[0].device

    def is_correct_device(column: ColumnProtocol) -> bool:
        return column.device == device

    if not all(map(is_correct_device, columns)):  # pragma: no cover
        msg = "Columns must be all on the same device."
        raise RuntimeError(msg)
    return device


def deduce_length(columns: Sequence[ColumnProtocol]) -> int:
    """
    Sanity check for matching lengths on all of dataset's columns.

    :param columns: A list of dataset's column data.
    :raises RuntimeError: If any of the columns has less rows than others.
    :return: The determined columns' length.
    """
    assert len(columns) > 0
    length = columns[0].length

    def is_correct_length(column: ColumnProtocol) -> bool:
        return column.length == length

    if not all(map(is_correct_length, columns)):
        msg = "Columns must have the same lengths."
        raise RuntimeError(msg)
    assert length > 0
    return length


def deduce_length_device(columns: dict[str, ColumnProtocol]) -> tuple[int, torch.device]:
    """A combination check for both matching devices and lengths."""
    raw = [*columns.values()]
    columns_length = deduce_length(raw)
    columns_device = deduce_device(raw)
    del raw
    return (columns_length, columns_device)


class NamedColumns:
    """
    Representation of a data batch read from the filesystem.
    This representation contains all of the columns read into memory, as well as
    metadata such as their length and current device.
    """

    def __init__(
        self,
        columns: dict[str, ColumnProtocol],
        make_mask_name: Callable[[str], str] = DEFAULT_MAKE_MASK_NAME,
    ) -> None:
        """
        :param columns: Column data read from the filesystem.
        :param make_mask_name: A function generating matching mask names for each column.
        """
        self.columns_length, self.columns_device = deduce_length_device(columns)

        self.columns = columns
        self.make_mask_name = make_mask_name

    @property
    def length(self) -> int:
        return self.columns_length

    @property
    def device(self) -> torch.device:
        return self.columns_device

    def __len__(self) -> int:
        return self.columns_length

    def __getitem__(self, indices: torch.LongTensor) -> Batch:
        indices = indices.to(device=self.device)
        result = {}
        for name, column in self.columns.items():
            result[self.make_mask_name(name)], result[name] = column[indices]
        return result
