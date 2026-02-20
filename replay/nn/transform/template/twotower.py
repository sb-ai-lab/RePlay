import torch

from replay.data.nn import TensorSchema

from .sasrec import make_default_sasrec_transforms


def make_default_twotower_transforms(tensor_schema: TensorSchema) -> dict[str, list[torch.nn.Module]]:
    """
    Creates a valid transformation pipeline for TwoTower data batches for usage in :ref:`Parquet-Module` .

    Generated pipeline expects input dataset to contain all features specified in the ``tensor_schema``.

    :param tensor_schema: TensorSchema used to infer feature columns.
    :return: dict of transforms specified for every dataset split (train, validation, test, predict).
    """
    return make_default_sasrec_transforms(tensor_schema=tensor_schema)
