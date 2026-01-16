import copy

from replay.data.nn import TensorSchema
from replay.nn.transforms import (
    BaseTransform,
    GroupTransform,
    NextTokenTransform,
    RenameTransform,
)


def make_default_sasrec_transforms(
    tensor_schema: TensorSchema, query_column: str = "query_id"
) -> dict[str, list[BaseTransform]]:
    """
    Generates a valid transformation pipeline for SasRec data batches.

    Genearted pipeline expects input dataset to contain the following columns:
        1) Query ID column, specified by ``query_column``.
        2) Item ID column, specified in the tensor schema.
        3) ``"train"`` - item IDs used for training (Validation/Test subsets only) .

    :param tensor_schema: TensorSchema used to infer feature columns.
    :param query_column: Name of the column containing query IDs. Default: ``"query_id"``.
    :param use_legacy: If ``True``, map batches to old model version's ``NamedTuple`` isntances.
        Default: ``False``.
    :return: _description_
    """
    item_column = tensor_schema.item_id_feature_name
    train_transforms = [
        NextTokenTransform(label_field=item_column, query_features=query_column, shift=1),
        RenameTransform(
            {query_column: "query_id", f"{item_column}_mask": "padding_mask", "labels_mask": "labels_padding_mask"}
        ),
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
