"""
Implementation of the ``ParquetDataset`` and its internals.

``ParquetDataset`` is combination of PyTorch-compatible dataset and sampler which enables
training and inference of models on datasets of any arbitrary size by leveraging PyArrow
Datasets to perform batch-wise reading and processing of data from disk.

``ParquetDataset`` includes support for Pytorch's distributed training framework as well as
access to remotely stored data via PyArrow's filesystem configs.
"""

from .info.replicas import DEFAULT_REPLICAS_INFO, ReplicasInfo, ReplicasInfoProtocol
from .parquet_dataset import ParquetDataset
from .parquet_module import ParquetModule

__all__ = [
    "DEFAULT_REPLICAS_INFO",
    "ParquetDataset",
    "ParquetModule",
    "ReplicasInfo",
    "ReplicasInfoProtocol",
]
