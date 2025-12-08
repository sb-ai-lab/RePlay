from functools import lru_cache

import numpy as np
import pyarrow as pa
import torch


def torch_to_numpy(dtype: torch.dtype) -> np.dtype:
    @lru_cache
    def _torch_to_numpy(dtype: torch.dtype) -> np.dtype:
        exemplar: torch.Tensor = torch.asarray([0], dtype=dtype)
        return exemplar.numpy().dtype

    return _torch_to_numpy(dtype)


def numpy_to_torch(dtype: np.dtype) -> torch.dtype:
    @lru_cache
    def _numpy_to_torch(dtype: np.dtype) -> torch.dtype:
        exemplar: np.ndarray = np.asarray([0], dtype=dtype)
        return torch.from_numpy(exemplar).dtype

    return _numpy_to_torch(dtype)


def pyarrow_to_numpy(dtype: pa.DataType) -> np.dtype:
    @lru_cache
    def _pyarrow_to_numpy(dtype: pa.DataType) -> np.dtype:
        exemplar: pa.Array = pa.array([0], type=dtype)
        return exemplar.to_numpy().dtype

    return _pyarrow_to_numpy(dtype)


def numpy_to_pyarrow(dtype: np.dtype) -> pa.DataType:
    @lru_cache
    def _numpy_to_pyarrow(dtype: np.dtype) -> pa.DataType:
        exemplar: np.ndarray = np.asarray([0], dtype=dtype)
        return pa.array(exemplar).type

    return _numpy_to_pyarrow(dtype)


def torch_to_pyarrow(dtype: torch.dtype) -> pa.DataType:
    @lru_cache
    def _torch_to_pyarrow(dtype: torch.dtype) -> pa.DataType:
        np_dtype: np.dtype = torch_to_numpy(dtype)
        return numpy_to_pyarrow(np_dtype)

    return _torch_to_pyarrow(dtype)


def pyarrow_to_torch(dtype: pa.DataType) -> torch.dtype:
    @lru_cache
    def _pyarrow_to_torch(dtype: pa.DataType) -> torch.dtype:
        np_dtype: np.dtype = pyarrow_to_numpy(dtype)
        return numpy_to_torch(np_dtype)

    return _pyarrow_to_torch(dtype)
