from collections.abc import Callable, Iterator
from typing import Any

import pyarrow.dataset as da
import torch

from replay.data.nn.parquet.constants.device import DEFAULT_DEVICE
from replay.data.nn.parquet.impl.masking import DEFAULT_MAKE_MASK_NAME

from .impl.array_1d_column import to_array_1d_columns
from .impl.array_2d_column import to_array_2d_columns
from .impl.named_columns import NamedColumns
from .impl.numeric_column import to_numeric_columns
from .metadata import Metadata


class BatchesIterator:
    """Iterator for batch-wise extraction of data from a Parquet dataset with conversion to structured columns."""

    def __init__(
        self,
        metadata: Metadata,
        dataset: da.Dataset,
        batch_size: int,
        make_mask_name: Callable[[str], str] = DEFAULT_MAKE_MASK_NAME,
        device: torch.device = DEFAULT_DEVICE,
        pyarrow_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        :param metadata: Metadata describing the structure and types of input data.
        :param dataset: Pyarrow dataset implementing the ``to_batches`` method.
        :param batch_size: Batch size sampled from a single partition.
            Resulting batch will not always match it in size due to mismatches between
            the target batch size and the partition size.
        :param make_mask_name: Mask name generation function. Default: value of ``DEFAULT_MAKE_MASK_NAME``.
        :param device: The device on which the data will be generated. Defaults: value of ``DEFAULT_DEVICE``.
        :param pyarrow_kwargs: Additional parameters for PyArrow dataset's ``to_batches`` method. Default: ``None``.
        """
        if pyarrow_kwargs is None:
            pyarrow_kwargs = {}
        self.dataset = dataset
        self.metadata = metadata
        self.batch_size = batch_size
        self.make_mask_name = make_mask_name
        self.device = device
        self.pyarrow_kwargs = pyarrow_kwargs

    def __iter__(self) -> Iterator[NamedColumns]:
        for batch in self.dataset.to_batches(
            batch_size=self.batch_size,
            columns=list(self.metadata.keys()),
            **self.pyarrow_kwargs,
        ):
            yield NamedColumns(
                columns={
                    **to_numeric_columns(batch, self.metadata, self.device),
                    **to_array_1d_columns(batch, self.metadata, self.device),
                    **to_array_2d_columns(batch, self.metadata, self.device),
                },
                make_mask_name=self.make_mask_name,
            )
