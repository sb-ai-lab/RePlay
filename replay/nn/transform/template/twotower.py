import torch

from replay.data.nn import TensorSchema

from .sasrec import make_default_sasrec_transforms


def make_default_twotower_transforms(
    tensor_schema: TensorSchema, query_column: str = "query_id"
) -> dict[str, list[torch.nn.Module]]:
    """
    Creates a valid transformation pipeline for TwoTower data batches.

    Generated pipeline expects input dataset to contain the following columns:
        1) Query ID column, specified by ``query_column``.
        2) Item ID column, specified in the tensor schema.

    :param tensor_schema: TensorSchema used to infer feature columns.
    :param query_column: Name of the column containing query IDs. Default: ``"query_id"``.
    :return: dict of transforms specified for every dataset split (train, validation, test, predict).
    """
    return make_default_sasrec_transforms(tensor_schema=tensor_schema, query_column=query_column)
