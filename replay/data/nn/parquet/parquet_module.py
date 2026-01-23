import copy
import warnings
from typing import Literal, Optional, get_args

import lightning as L  # noqa: N812
import torch
from lightning.pytorch.trainer.states import RunningStage
from typing_extensions import TypeAlias, override

from replay.data.nn.parquet.constants.filesystem import DEFAULT_FILESYSTEM
from replay.data.nn.parquet.impl.masking import (
    DEFAULT_COLLATE_FN,
    DEFAULT_MAKE_MASK_NAME,
    DEFAULT_REPLICAS_INFO,
)
from replay.data.nn.parquet.parquet_dataset import ParquetDataset

TransformStage: TypeAlias = Literal["train", "validate", "test", "predict"]

DEFAULT_CONFIG = {"train": {"generator": torch.default_generator}}


class ParquetModule(L.LightningDataModule):
    """
    Standardized DataModule with batch-wise support via `ParquetDataset`.

    Allows for unified access to all data splits across the training/inference pipeline without loading
    full dataset into memory. See the :ref:`parquet-processing` section for details.

    ParquetModule provides per batch data loading and preprocessing via transform pipelines.
    See the :ref:`Transforms` section for getting info about available batch transforms.

    **Note:**

    *   ``ParquetModule`` supports only numeric values (boolean/integer/float),
        therefore, the data paths passed as arguments must contain encoded data.
    *   For optimal performance, set the OMP_NUM_THREADS and ARROW_IO_THREADS to match
        the number of available CPU cores.
    *   It's possible to use all train/validate/test/predict splits, then paths to splits should be passed
        as corresponding arguments of ``ParquetModule``.
        Alternatively, all the paths to the splits may be not specified
        but then do not forget to configure the Pytorch Lightning Trainer's instance accordingly.
        For example, if you don't want use validation data, you are able not to set ``validate_path`` parameter
        in ``ParquetModule`` and set ``limit_val_batches=0`` in Ligthning.Trainer.

    """

    def __init__(
        self,
        batch_size: int,
        metadata: dict,
        transforms: dict[TransformStage, list[torch.nn.Module]],
        config: Optional[dict] = None,
        *,
        train_path: Optional[str] = None,
        validate_path: Optional[str] = None,
        test_path: Optional[str] = None,
        predict_path: Optional[str] = None,
    ) -> None:
        """
        :param batch_size: Target batch size.
        :param metadata: A dictionary that each data split maps to a dictionary of feature names
            with each feature is associated with its shape and padding_value.\n
            Example: {"train": {"item_id" : {"shape": 100, "padding_value": 7657}}}.\n
            For details, see the section :ref:`parquet-processing`.
        :param config: Dict specifying configuration options of ``ParquetDataset`` (generator,
            filesystem, collate_fn, make_mask_name, replicas_info) for each data split.
            Default: ``DEFAULT_CONFIG``.\n
            In most scenarios, the default configuration is sufficient.
        :param transforms: Dict specifying sequence of Transform modules for each data split.
        :param train_path: Path to the Parquet file containing train data split. Default: ``None``.
        :param validate_path: Path to the Parquet file containing validation data split. Default: ``None``.
        :param test_path: Path to the Parquet file containing testing data split. Default: ``None``.
        :param predict_path: Path to the Parquet file containing prediction data split. Default: ``None``.
        """
        if not any([train_path, validate_path, test_path, predict_path]):
            msg = (
                f"{type(self)}.__init__() expects at least one of "
                "['train_path', 'val_path', 'test_path', 'predict_path], but none were provided."
            )
            raise KeyError(msg)

        super().__init__()
        if config is None:
            config = DEFAULT_CONFIG

        self.datapaths = {"train": train_path, "validate": validate_path, "test": test_path, "predict": predict_path}
        missing_splits = [split_name for split_name, split_path in self.datapaths.items() if split_path is None]
        if missing_splits:
            msg = (
                f"The following dataset paths aren't provided: {','.join(missing_splits)}."
                "Make sure to disable these stages in your Lightning Trainer configuration."
            )
            warnings.warn(msg, stacklevel=2)

        self.metadata = copy.deepcopy(metadata)
        self.batch_size = batch_size
        self.config = config

        self.transforms = transforms
        self.compiled_transforms = self.prepare_transforms(transforms)

    def prepare_transforms(
        self, transforms: dict[TransformStage, list[torch.nn.Module]]
    ) -> dict[TransformStage, torch.nn.Sequential]:
        """
        Preform meta adjustments based on provided transform pipelines,
        then compile each subset into a `torch.nn.Sequential` module.

        :param: transforms: Python dict where keys are names of stage (train, validate, test, predict)
            and values are corresponding transform pipelines for every stage.
        :returns: out: Compiled transform pipelines.
        """
        if not any(subset in get_args(TransformStage) for subset in transforms):
            msg = (
                f"Expected transform.keys()={list(transforms.keys())} to contain at least "
                f"one of {get_args(TransformStage)}, but none were found."
            )
            raise KeyError(msg)

        compiled_transorms = {}
        for subset, transform_set in transforms.items():
            compiled_transorms[subset] = torch.nn.Sequential(*transform_set)

        return compiled_transorms

    @override
    def setup(self, stage):
        self.datasets = {}

        for subset in get_args(TransformStage):
            if self.datapaths.get(subset, None) is not None:
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
        return self.datasets["validate"]

    @override
    def test_dataloader(self):
        return self.datasets["test"]

    @override
    def predict_dataloader(self):
        return self.datasets["predict"]

    @override
    def on_after_batch_transfer(self, batch, _dataloader_idx):
        stage = self.trainer.state.stage
        target = RunningStage.VALIDATING if stage is RunningStage.SANITY_CHECKING else stage

        return self.compiled_transforms[str(target.value)](batch)
