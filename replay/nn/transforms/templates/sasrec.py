import copy

import torch

from replay.data.nn import TensorSchema
from replay.nn.transforms import GroupTransform, NextTokenTransform, RenameTransform, UnsqueezeTransform


def make_default_sasrec_transforms(
    tensor_schema: TensorSchema, query_column: str = "query_id"
) -> dict[str, list[torch.nn.Module]]:
    """
    Creates a valid transformation pipeline for SasRec data batches.

    Generated pipeline expects input dataset to contain the following columns:
        1) Query ID column, specified by ``query_column``.
        2) Item ID column, specified in the tensor schema.

    :param tensor_schema: TensorSchema used to infer feature columns.
    :param query_column: Name of the column containing query IDs. Default: ``"query_id"``.
    :return: dict of transforms specified for every dataset split (train, validation, test, predict).
    """
    item_column = tensor_schema.item_id_feature_name
    train_transforms = [
        NextTokenTransform(label_field=item_column, query_features=query_column, shift=1),
        RenameTransform(
            {
                query_column: "query_id",
                f"{item_column}_mask": "padding_mask",
                "positive_labels_mask": "target_padding_mask",
            }
        ),
        UnsqueezeTransform("target_padding_mask", -1),
        UnsqueezeTransform("positive_labels", -1),
        GroupTransform({"feature_tensors": [item_column]}),
    ]

    val_transforms = [
        RenameTransform({query_column: "query_id", f"{item_column}_mask": "padding_mask"}),
        GroupTransform({"feature_tensors": [item_column]}),
    ]
    test_transforms = copy.deepcopy(val_transforms)

    predict_transforms = copy.deepcopy(val_transforms)

    transforms = {
        "train": train_transforms,
        "validate": val_transforms,
        "test": test_transforms,
        "predict": predict_transforms,
    }

    return transforms
