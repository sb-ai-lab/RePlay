from functools import lru_cache

import numpy as np
import pyarrow as pa
import torch


@lru_cache
def _torch_to_numpy(dtype: torch.dtype) -> np.dtype:
    exemplar: torch.Tensor = torch.asarray([0], dtype=dtype)
    return exemplar.numpy().dtype


def torch_to_numpy(dtype: torch.dtype) -> np.dtype:
    return _torch_to_numpy(dtype)


@lru_cache
def _numpy_to_torch(dtype: np.dtype) -> torch.dtype:
    exemplar: np.ndarray = np.asarray([0], dtype=dtype)
    return torch.from_numpy(exemplar).dtype


def numpy_to_torch(dtype: np.dtype) -> torch.dtype:
    return _numpy_to_torch(dtype)


@lru_cache
def _pyarrow_to_numpy(dtype: pa.DataType) -> np.dtype:
    exemplar: pa.Array = pa.array([0], type=dtype)
    return exemplar.to_numpy().dtype


def pyarrow_to_numpy(dtype: pa.DataType) -> np.dtype:
    return _pyarrow_to_numpy(dtype)


@lru_cache
def _numpy_to_pyarrow(dtype: np.dtype) -> pa.DataType:
    exemplar: np.ndarray = np.asarray([0], dtype=dtype)
    return pa.array(exemplar).type


def numpy_to_pyarrow(dtype: np.dtype) -> pa.DataType:
    return _numpy_to_pyarrow(dtype)


@lru_cache
def _torch_to_pyarrow(dtype: torch.dtype) -> pa.DataType:
    np_dtype: np.dtype = torch_to_numpy(dtype)
    return numpy_to_pyarrow(np_dtype)


def torch_to_pyarrow(dtype: torch.dtype) -> pa.DataType:
    return _torch_to_pyarrow(dtype)


@lru_cache
def _pyarrow_to_torch(dtype: pa.DataType) -> torch.dtype:
    np_dtype: np.dtype = pyarrow_to_numpy(dtype)
    return numpy_to_torch(np_dtype)


def pyarrow_to_torch(dtype: pa.DataType) -> torch.dtype:
    return _pyarrow_to_torch(dtype)
