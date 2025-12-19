import copy
import warnings
from typing import Literal, Optional, get_args

import lightning as L  # noqa: N812
import torch
from typing_extensions import TypeAlias, override

from replay.data.nn.parquet.constants.filesystem import DEFAULT_FILESYSTEM
from replay.data.nn.parquet.impl.masking import (
    DEFAULT_COLLATE_FN,
    DEFAULT_MAKE_MASK_NAME,
    DEFAULT_REPLICAS_INFO,
)
from replay.data.nn.parquet.parquet_dataset import ParquetDataset
from replay.nn.transforms.base import BaseTransform

TransformStage: TypeAlias = Literal["train", "val", "test"]

DEFAULT_CONFIG = {"train": {"generator": torch.default_generator}}


class ParquetModule(L.LightningDataModule):
    """
    Standardized DataModule with batch-wise support via `ParquetDataset`.

    Allows for unified access to all data splits across the training/inference pipeline without loading
    full dataset into memory. Provide per batch data loading and preprocessing via transform pipelines.
    See the :ref:`Transforms` section for getting info about available batch transforms.

    It's possible to use all train/val/test splits, then paths to splits should be passed
    as corresponding arguments of `ParquetModule`.
    Alternatively, all the paths to the splits may be not specified
    but then do not forget to configure the Pytorch Lightning Trainer's instance accordingly.

    For example, if you don't want use validation data, you are able not to set ``val_path`` parameter
    in `ParquetModule` and set ``limit_val_batches=0`` in Ligthning.Trainer.
    """

    def __init__(
        self,
        batch_size: int,
        metadata: dict,
        transforms: dict[TransformStage, list[torch.nn.Module]],
        config: dict = DEFAULT_CONFIG,
        *,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
    ):
        """
        :param batch_size: Target batch size.
        :param metadata: A dictionary containing information about the data columns to be read from the parquet files
            for each split.
        :param config: Dict specifying configuration options, such as `DataLoader` generators,
            filesystem, collate function, etc. for each data split.
        :param transforms: Dict specifying sequence of Transform modules for each data split.
        :param train_path: Path to the Parquet file containing train data split. Default: `None`.
        :param val_path: Path to the Parquet file containing validation data split. Default: `None`.
        :param test_path: Path to the Parquet file containing test data split. Default: `None`.
        """
        if not any([train_path, val_path, test_path]):
            msg = (
                f"{type(self)}.__init__() expects at least one of ['train_path', 'val_path', 'test_path'] "
                "but none of them founded."
            )
            raise KeyError(msg)

        super().__init__()
        self.datapaths = {"train": train_path, "val": val_path, "test": test_path}
        missing_splits = [split_name for split_name, split_path in self.datapaths.items() if split_path is None]
        if missing_splits:
            msg = (
                f"The following dataset paths aren't provided: {','.join(missing_splits)}."
                "Make sure to disable these splits in your Lightning Trainer configuration."
            )
            warnings.warn(msg, stacklevel=2)

        self.metadata = copy.deepcopy(metadata)
        self.batch_size = batch_size
        self.config = config

        self.transforms = transforms
        self.compiled_transforms = self.prepare_transforms(transforms)

    def prepare_transforms(
        self, transforms: dict[TransformStage, list[BaseTransform]]
    ) -> dict[TransformStage, torch.nn.Sequential]:
        """
        Preform meta adjustments based on provided transform pipelines,
        then compile each subset into a `torch.nn.Sequential` module.

        :param: transforms: Python dict where keys are names of stage (train, val, test) and values are
            corresponding transform pipelines for every stage.
        :returns: out: Compiled transform pipelines.
        """
        if not any(subset in get_args(TransformStage) for subset in transforms):
            msg = f"`transforms` expects at least one of {get_args(TransformStage)}, but none were found."
            raise KeyError(msg)

        compiled_transorms = {}
        for subset, transform_set in transforms.items():
            for transform in transform_set:
                self.metadata[subset] = transform.adjust_meta(self.metadata[subset])

            compiled_transorms[subset] = torch.nn.Sequential(*transform_set)

        return compiled_transorms

    @override
    def setup(self, stage):
        self.datasets = {}

        for subset in get_args(TransformStage):
            if self.datapaths[subset] is not None:
                subset_config = self.config.get(subset, {})
                kwargs = {
                    "source": self.datapaths[subset],
                    "metadata": self.metadata[subset],
                    "batch_size": self.batch_size,
                    "partition_size": subset_config.get("partition_size", 2**17),
                    "generator": subset_config.get("generator", None),
                    "filesystem": subset_config.get("filesystem", DEFAULT_FILESYSTEM),
                    "make_mask_name": subset_config.get("make_mask_name", DEFAULT_MAKE_MASK_NAME),
                    "replicas_info": subset_config.get("replicas_info", DEFAULT_REPLICAS_INFO),
                    "collate_fn": subset_config.get("collate_fn", DEFAULT_COLLATE_FN),
                }

                self.datasets[subset] = ParquetDataset(**kwargs)

    @override
    def train_dataloader(self):
        return self.datasets["train"]

    @override
    def val_dataloader(self):
        return self.datasets["val"]

    @override
    def predict_dataloader(self):
        return self.datasets["test"]

    @override
    def on_after_batch_transfer(self, batch, _dataloader_idx):
        subset = "test"
        if self.trainer.training:
            subset = "train"
        elif not self.trainer.predicting:
            subset = "val"

        return self.compiled_transforms[subset](batch)
