import pytest

from replay.data.nn.module.datamodule import ParquetModule


def test_datamodule_raises_no_paths_provided(parquet_module_args):
    with pytest.raises(KeyError):
        ParquetModule(**parquet_module_args)


def test_datamodule_raises_no_transforms_provided(parquet_module_path, parquet_module_args):
    parquet_module_args["transforms"] = {}
    with pytest.raises(KeyError):
        ParquetModule(train_path=parquet_module_path, **parquet_module_args)
