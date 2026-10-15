import copy

import torch

from replay.data.nn import TensorSchema
from replay.nn.transform import (
    CopyTransform,
    GroupTransform,
    RenameTransform,
    SequenceRollTransform,
    TargetMLMTransform,
    TokenMaskTransform,
    UnsqueezeTransform,
)


def make_default_bert4rec_transforms(tensor_schema: TensorSchema) -> dict[str, list[torch.nn.Module]]:
    """
    Creates a valid transformation pipeline for BERT4Rec data batches for usage in :ref:`Parquet-Module` .

    The generated pipeline expects an input dataset to contain all features specified in the ``tensor_schema``.

    :param tensor_schema: TensorSchema used to infer feature columns.
    :return: a dict of transforms specified for every dataset split (train, validation, test, predict).
    """
    item_id_name = tensor_schema.item_id_feature_name
    padding_mask_name = f"{item_id_name}_mask"

    train_transforms = [
        RenameTransform({padding_mask_name: "padding_mask"}),
        TokenMaskTransform(token_field="padding_mask"),
        CopyTransform({item_id_name: "positive_labels"}),
        TargetMLMTransform(),
        UnsqueezeTransform("target_padding_mask", -1),
        UnsqueezeTransform("positive_labels", -1),
        GroupTransform({"feature_tensors": [*tensor_schema.names, "token_mask"]}),
    ]

    inference_transforms = [
        RenameTransform({padding_mask_name: "padding_mask"}),
        CopyTransform({"padding_mask": "token_mask"}),
        *[
            SequenceRollTransform(feature_name, roll=-1, padding_value=feature.padding_value)
            for feature_name, feature in tensor_schema.items()
            if feature.is_seq
        ],
        SequenceRollTransform("padding_mask", roll=-1, padding_value=1),
        SequenceRollTransform("token_mask", roll=-1, padding_value=0),
        GroupTransform({"feature_tensors": [*tensor_schema.names, "token_mask"]}),
    ]
    val_transforms = copy.deepcopy(inference_transforms)
    test_transforms = copy.deepcopy(inference_transforms)
    predict_transforms = copy.deepcopy(inference_transforms)

    transforms = {
        "train": train_transforms,
        "validate": val_transforms,
        "test": test_transforms,
        "predict": predict_transforms,
    }

    return transforms
