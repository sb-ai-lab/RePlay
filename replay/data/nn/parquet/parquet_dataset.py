import warnings
from collections.abc import Callable, Iterator
from typing import Optional, Union, cast

import pyarrow.dataset as ds
import pyarrow.fs as fs
import torch
from torch.utils.data import IterableDataset

from replay.data.nn.parquet.constants.batches import GeneralBatch
from replay.data.nn.parquet.constants.device import DEFAULT_DEVICE
from replay.data.nn.parquet.constants.filesystem import DEFAULT_FILESYSTEM
from replay.data.nn.parquet.impl.masking import (
    DEFAULT_COLLATE_FN,
    DEFAULT_MAKE_MASK_NAME,
    DEFAULT_REPLICAS_INFO,
    GeneralCollateFn,
)
from replay.data.nn.parquet.info.replicas import ReplicasInfoProtocol
from replay.data.nn.parquet.utils.compute_length import compute_fixed_size_length

from .fixed_batch_dataset import FixedBatchSizeDataset
from .iterator import BatchesIterator
from .metadata import Metadata
from .partitioned_iterable_dataset import PartitionedIterableDataset


class ParquetDataset(IterableDataset):
    """
    Combination dataset and sampler for batch-wise reading and processing of Parquet files.

    This implementation allows one to read data using a PyArrow Dataset, convert it into structured columns,
    split it into partitions, and then into batches needed for model training.
    Supports distributed training and reproducible random shuffling.

    During data loader operation, a partition of size ``partition_size`` is read.
    There may be situations where the size of the read partition is less than
    `partition_size` - this depends on the number of rows in the data fragment.
    A fragment is a single Parquet file in the file system.

    The read partition will be processed and the result will be returned as a batch of size ``batch_size``.
    Please note that the resulting batch size may be less than ``batch_size``.

    For maximum efficiency when reading and processing data,
    it is recommended to set `partition_size` to several times larger than `batch_size`.
    """

    def __init__(
        self,
        source: Union[str, list[str]],
        metadata: Metadata,
        partition_size: int,
        batch_size: int,
        filesystem: Union[str, fs.FileSystem] = DEFAULT_FILESYSTEM,
        make_mask_name: Callable[[str], str] = DEFAULT_MAKE_MASK_NAME,
        device: torch.device = DEFAULT_DEVICE,
        generator: Optional[torch.Generator] = None,
        replicas_info: ReplicasInfoProtocol = DEFAULT_REPLICAS_INFO,
        collate_fn: GeneralCollateFn = DEFAULT_COLLATE_FN,
        **kwargs,
    ) -> None:
        """
        :param source: The path or list of paths to files/directories containing data in Parquet format.
        :param metadata: Metadata describing the data structure.
            The structure of each column is defined by the following values:

                ``shape`` - the dimension of the column being read.
                    If the column contains only one value, this parameter does not need to be specified.
                    If the column contains a one-dimensional array, the parameter must be a number or an array
                    containing one number.
                    If the column contains a two-dimensional array, the parameter
                    must be an array containing two numbers.

                ``padding`` - padding value that will fill the arrays if their length is less
                    than that specified in the `shape` parameter.
        :param partition_size: Partition size when reading data from Parquet files.
        :param batch_size: The size of the batch that will be returned during iteration.
        :param filesystem: A PyArrow's Filesystem object used to access data, or a URI-based path
            to infer the filesystem from. Default: value of ``DEFAULT_FILESYSTEM``.
        :param make_mask_name: Mask name generation function. Default: value of ``DEFAULT_MAKE_MASK_NAME``.
        :param device: The device on which the data will be generated. Defaults: value of ``DEFAULT_DEVICE``.
        :param generator: Random number generator for batch shuffling.
            If ``None``, shuffling will be disabled. Default: ``None``.
        :param replicas_info: A replica info object capable of fetching information about the distributed environment.
            Default: value of ``DEFAULT_REPLICAS_INFO`` - a default wrapper utilizing functions from the
            ``torch.utils.data`` and ``torch.distributed`` modules.
        :param collate_fn: Collate function for merging batches. Default: value of ``DEFAULT_COLLATE_FN``.
        """
        if partition_size < batch_size:
            msg = (
                "Suboptimal parameters: partition size is smaller than batch size. "
                f"Got: {partition_size=}, {batch_size=}."
            )
            warnings.warn(msg, stacklevel=2)

        if (partition_size % batch_size) != 0:
            msg = (
                "Suboptimal parameters: partition size is not multiple of batch size. "
                f"Got: {partition_size=}, {batch_size=}."
            )
            warnings.warn(msg, stacklevel=2)

        if isinstance(filesystem, str):
            filesystem, _ = fs.FileSystem.from_uri(filesystem)
        assert isinstance(filesystem, fs.FileSystem)
        self.filesystem = cast(fs.FileSystem, filesystem)

        self.pyarrow_dataset = ds.dataset(
            source,
            filesystem=self.filesystem,
            format="parquet",
            **kwargs.get("pyarrow_dataset_kwargs", {}),
        )

        self.batch_size = batch_size
        self.partition_size = partition_size
        self.replicas_info = replicas_info
        self.metadata = metadata

        self.iterator = BatchesIterator(
            dataset=self.pyarrow_dataset,
            metadata=self.metadata,
            batch_size=partition_size,
            device=device,
            make_mask_name=make_mask_name,
            pyarrow_kwargs=kwargs.get("pyarrow_to_batches_kwargs", {}),
        )

        self.raw_dataset = PartitionedIterableDataset(
            batch_size=batch_size,
            iterable=self.iterator,
            generator=generator,
            replicas_info=replicas_info,
        )

        self.dataset = FixedBatchSizeDataset(
            dataset=self.raw_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        self.do_compute_length = True
        self.cached_lengths: dict[int, int] = {}

    def compute_length(self) -> int:
        """Returns the length of the dataset counted in fixed-size batches."""
        num_replicas = self.replicas_info.num_replicas
        if num_replicas not in self.cached_lengths:
            if len(self.cached_lengths) > 0:
                msg = "`num_replicas` changed. Unable to reuse cached length."
                warnings.warn(msg, stacklevel=2)
            curr_length = compute_fixed_size_length(
                iterable=self.iterator,
                num_replicas=num_replicas,
                batch_size=self.batch_size,
            )
            self.cached_lengths[num_replicas] = curr_length
        return self.cached_lengths[num_replicas]

    def __len__(self) -> int:
        if self.do_compute_length:
            return self.compute_length()
        msg = "This instance doesn't support `len()` method. You can enable it by setting `do_compute_length=True`."
        raise TypeError(msg)

    def __iter__(self) -> Iterator[GeneralBatch]:
        return iter(self.dataset)
