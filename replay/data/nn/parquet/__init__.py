"""
Implementation of the ``ParquetDataset`` and its internals.

``ParquetDataset`` is combination of PyTorch-compatible dataset and sampler which enables
training and inference of models on datasets of any arbitrary size by leveraging PyArrow
Datasets to perform batch-wise reading and processing of data from disk.

``ParquetDataset`` includes support for Pytorch's distributed training framework as well as
access to remotely stored data via PyArrow's filesystem configs.
"""

from .parquet_dataset import ParquetDataset
from .info.replicas import ReplicasInfoProtocol, ReplicasInfo, DEFAULT_REPLICAS_INFO

__all__ = [
    "DEFAULT_REPLICAS_INFO",
    "ParquetDataset",
    "ReplicasInfo",
    "ReplicasInfoProtocol",
]
