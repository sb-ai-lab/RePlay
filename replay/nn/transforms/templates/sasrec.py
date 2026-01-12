from replay.models.nn.sequential.sasrec.dataset import SasRecPredictionBatch, SasRecTrainingBatch, SasRecValidationBatch
from replay.nn.transforms import (
    BaseTransform,
    BatchingTransform,
    CopyTransform,
    GroupTransform,
    NextTokenTransform,
    RenameTransform,
)


# TODO: Offer func names
# TODO: List options for `query_column`
def make_sasrec_transforms(
    query_column: str = "query_id",
    item_column: str = "item_id", # TensorSchema?
    use_legacy: bool = False
) -> dict[str, list[BaseTransform]]:
    """
    Generates a valid transformation pipeline for SasRec data batches.

    Genearted pipeline expects input dataset to contain the following columns:
        1) Query ID column, specified by ``query_column``.
        2) Item ID column, specified by ``item_column``.
        3) ``"train"`` - item IDs used for training (Validation subset only) .

    :param query_column: Name of the column containing query IDs. Default: ``"query_id"``.
    :param item_column: Name of the column containing item IDs. Default: ``"item_id"``.
    :param use_legacy: If ``True``, map batches to old model version's ``NamedTuple`` isntances.
        Default: ``False``.
    :return: _description_
    """
    train_transforms = [
        NextTokenTransform(label_field=item_column, query_features=query_column, shift=1),
        RenameTransform({
            query_column: "query_id",
            f"{item_column}_mask": "padding_mask",
            "labels_mask": "labels_padding_mask"
        }),
        GroupTransform({"features": [item_column]}), # fetch from TensorSchema
    ]

    val_transforms = [
        RenameTransform({query_column: "query_id", f"{item_column}_mask": "padding_mask"}),
        CopyTransform(mapping={"train": "ground_truth"}), # Add mention of pre-made`train` to docs
        GroupTransform({"features": [item_column]}),
    ]

    test_transforms = [
        RenameTransform({query_column: "query_id", f"{item_column}_mask": "padding_mask"}),
        GroupTransform({"features": [item_column]}),
    ]

    if use_legacy:
        train_transforms.append(BatchingTransform(SasRecTrainingBatch))
        val_transforms.append(BatchingTransform(SasRecValidationBatch))
        test_transforms.append(BatchingTransform(SasRecPredictionBatch))

    transforms = {
        "train": train_transforms,
        "val": val_transforms,
        "test": test_transforms,
    }

    return transforms
