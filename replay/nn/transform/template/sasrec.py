import copy

import torch

from replay.data.nn import TensorSchema
from replay.nn.transform import GroupTransform, NextTokenTransform, RenameTransform, UnsqueezeTransform


def make_default_sasrec_transforms(tensor_schema: TensorSchema) -> dict[str, list[torch.nn.Module]]:
    """
    Creates a valid transformation pipeline for SasRec data batches.

    Generated pipeline expects input dataset to contain all features specified in the ``tensor_schema``.

    :param tensor_schema: TensorSchema used to infer feature columns.
    :return: dict of transforms specified for every dataset split (train, validation, test, predict).
    """
    item_column = tensor_schema.item_id_feature_name
    train_transforms = [
        NextTokenTransform(label_field=item_column, shift=1),
        RenameTransform({f"{item_column}_mask": "padding_mask", "positive_labels_mask": "target_padding_mask"}),
        UnsqueezeTransform("target_padding_mask", -1),
        UnsqueezeTransform("positive_labels", -1),
        GroupTransform({"feature_tensors": tensor_schema.names}),
    ]

    val_transforms = [
        RenameTransform({f"{item_column}_mask": "padding_mask"}),
        GroupTransform({"feature_tensors": tensor_schema.names}),
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
