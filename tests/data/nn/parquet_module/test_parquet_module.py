import copy

import pytest

from replay.data.nn import ParquetModule
from replay.data.nn.parquet import ParquetDataset


def test_datamodule_raises_no_paths_provided(parquet_module_args):
    with pytest.raises(KeyError):
        ParquetModule(**parquet_module_args)


def test_datamodule_invalid_train_path_provided(parquet_module_path: str, parquet_module_args) -> None:
    with pytest.raises(TypeError):
        ParquetModule(train_path=[parquet_module_path, parquet_module_path], **parquet_module_args)


def test_datamodule_raises_no_transforms_provided(parquet_module_path: str, parquet_module_args):
    parquet_module_args_copy = copy.deepcopy(parquet_module_args)
    parquet_module_args_copy |= {"transforms": {}}
    with pytest.raises(KeyError):
        ParquetModule(train_path=parquet_module_path, **parquet_module_args_copy)


def test_datamodule_test_access(parquet_module_path, parquet_module_args):
    module = ParquetModule(test_path=parquet_module_path, **parquet_module_args)
    module.setup(None)
    assert isinstance(module.test_dataloader(), ParquetDataset)
