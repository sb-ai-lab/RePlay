from typing import List

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from replay.data import FeatureHint, FeatureType
from replay.utils import TORCH_AVAILABLE, MissingImportType

if TORCH_AVAILABLE:
    import torch
    from replay.data.nn import (
        PandasSequentialDataset,
        SequentialDataset,
        TensorFeatureInfo,
        TorchSequentialDataset,
        TorchSequentialValidationDataset,
    )
    from replay.experimental.nn.data.schema_builder import TensorSchemaBuilder
else:
    PandasSequentialDataset = MissingImportType
    SequentialDataset = MissingImportType
    TorchSequentialDataset = MissingImportType


@pytest.mark.torch
def test_can_get_padded_sequence(sequential_dataset: SequentialDataset):
    sd = TorchSequentialDataset(sequential_dataset, max_sequence_length=3, padding_value=-1)

    assert len(sd) == 4

    _compare_sequence(sd, 0, "item_id", [-1, 0, 1])
    _compare_sequence(sd, 1, "item_id", [0, 2, 3])
    _compare_sequence(sd, 2, "item_id", [-1, -1, 1])
    _compare_sequence(sd, 3, "item_id", [3, 4, 5])

    _compare_sequence(sd, 0, "some_item_feature", [-1, 1, 2])
    _compare_sequence(sd, 1, "some_item_feature", [1, 3, 4])
    _compare_sequence(sd, 2, "some_item_feature", [-1, -1, 2])
    _compare_sequence(sd, 3, "some_item_feature", [4, 5, 6])


@pytest.mark.torch
def test_can_get_windowed_sequence(sequential_dataset: SequentialDataset):
    sd = TorchSequentialDataset(
        sequential_dataset,
        max_sequence_length=3,
        sliding_window_step=2,
        padding_value=-1,
    )

    assert len(sd) == 6

    _compare_sequence(sd, 0, "item_id", [-1, 0, 1])
    _compare_sequence(sd, 1, "item_id", [0, 2, 3])
    _compare_sequence(sd, 2, "item_id", [-1, -1, 1])
    _compare_sequence(sd, 3, "item_id", [3, 4, 5])
    _compare_sequence(sd, 4, "item_id", [1, 2, 3])
    _compare_sequence(sd, 5, "item_id", [0, 1, 2])


@pytest.mark.torch
def test_can_get_query_id(sequential_dataset: SequentialDataset):
    sd = TorchSequentialDataset(
        sequential_dataset,
        max_sequence_length=3,
        padding_value=-1,
    )

    for i in range(0, 4):
        assert sd[i][0].item() == i


@pytest.mark.torch
def test_can_get_query_id_windowed(sequential_dataset: SequentialDataset):
    sd = TorchSequentialDataset(
        sequential_dataset,
        max_sequence_length=3,
        sliding_window_step=2,
        padding_value=-1,
    )

    for i in range(0, 4):
        assert sd[i][0].item() == i

    assert sd[4][0].item() == i
    assert sd[5][0].item() == i


@pytest.mark.torch
def test_can_get_query_feature(sequential_dataset: SequentialDataset):
    sd = TorchSequentialDataset(
        sequential_dataset,
        max_sequence_length=3,
        padding_value=-1,
    )

    _compare_query_feature(sd, 0, "some_user_feature", 1)
    _compare_query_feature(sd, 1, "some_user_feature", 2)
    _compare_query_feature(sd, 2, "some_user_feature", 3)
    _compare_query_feature(sd, 3, "some_user_feature", 4)


@pytest.mark.torch
def test_can_get_windowed_query_feature(sequential_dataset: SequentialDataset):
    sd = TorchSequentialDataset(
        sequential_dataset,
        max_sequence_length=3,
        sliding_window_step=2,
        padding_value=-1,
    )

    _compare_query_feature(sd, 0, "some_user_feature", 1)
    _compare_query_feature(sd, 1, "some_user_feature", 2)
    _compare_query_feature(sd, 2, "some_user_feature", 3)
    _compare_query_feature(sd, 3, "some_user_feature", 4)
    _compare_query_feature(sd, 4, "some_user_feature", 4)
    _compare_query_feature(sd, 5, "some_user_feature", 4)


@pytest.mark.torch
def test_num_dtype(sequential_dataset, some_num_tensor_feature):
    feature = TensorFeatureInfo(name="user_id", feature_type=FeatureType.NUMERICAL, tensor_dim=64)
    assert (
        TorchSequentialDataset(
            sequential_dataset,
            max_sequence_length=3,
            sliding_window_step=2,
            padding_value=-1,
        )._get_tensor_dtype(feature)
        == torch.float32
    )

    feature._feature_type = None

    with pytest.raises(AssertionError):
        TorchSequentialDataset(
            sequential_dataset,
            max_sequence_length=3,
            sliding_window_step=2,
            padding_value=-1,
        )._get_tensor_dtype(feature)


@pytest.mark.torch
def test_label_not_in_ground_truth(sequential_dataset):
    with pytest.raises(ValueError) as exc:
        TorchSequentialValidationDataset(
            sequential_dataset, sequential_dataset, sequential_dataset, 5, label_feature_name="fake_user_id"
        )

    assert str(exc.value) == "Label feature name not found in ground truth schema"


@pytest.mark.torch
def test_label_not_in_train_dataset(sequential_dataset, item_user_sequential_dataset):
    with pytest.raises(ValueError) as exc:
        TorchSequentialValidationDataset(
            sequential_dataset,
            sequential_dataset,
            item_user_sequential_dataset,
            5,
            label_feature_name="some_item_feature",
        )

    assert str(exc.value) == "Label feature name not found in train schema"


@pytest.mark.torch
@pytest.mark.parametrize(
    "feature_name, exception_msg",
    [
        ("some_item_feature", "Label feature must be categorical"),
        ("some_user_feature", "Label feature must be sequential"),
    ],
)
def test_feature_is_not_categorical(wrong_sequential_dataset, feature_name, exception_msg):
    with pytest.raises(ValueError) as exc:
        TorchSequentialValidationDataset(
            wrong_sequential_dataset,
            wrong_sequential_dataset,
            wrong_sequential_dataset,
            5,
            label_feature_name=feature_name,
        )

    assert str(exc.value) == exception_msg


@pytest.mark.torch
def test_common_query_ids(sequential_dataset):
    sequences = pd.DataFrame(
        [
            (4, np.array([0, 1, 1, 1, 2])),
            (5, np.array([0, 1, 3, 1, 2])),
            (6, np.array([0, 2, 3, 1, 2])),
            (7, np.array([1, 2, 0, 1, 2])),
        ],
        columns=[
            "user_id",
            "item_id",
        ],
    )

    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_hint=FeatureHint.ITEM_ID,
        )
        .build()
    )

    new_dataset = PandasSequentialDataset(
        tensor_schema=schema,
        query_id_column="user_id",
        item_id_column="item_id",
        sequences=sequences,
    )

    with pytest.raises(ValueError) as exc1:
        TorchSequentialValidationDataset(
            sequential_dataset,
            new_dataset,
            new_dataset,
            5,
        )

    with pytest.raises(ValueError) as exc2:
        TorchSequentialValidationDataset(
            sequential_dataset,
            sequential_dataset,
            new_dataset,
            5,
        )

    assert str(exc1.value) == "Sequential data and ground truth must contain the same query IDs"
    assert str(exc2.value) == "Sequential data and train must contain the same query IDs"


@pytest.mark.torch
@pytest.mark.parametrize(
    "feature_name, cardinality, exception_msg",
    [
        ("item_id_feature", 4, "Schema mismatch: item feature name does not match ground truth"),
        ("item_id", 5, "Schema mismatch: item feature cardinality does not match ground truth"),
    ],
)
def test_schemes_mismatch(tensor_schema, feature_name, cardinality, exception_msg):
    tensor_schema_gt = (
        TensorSchemaBuilder()
        .categorical(
            feature_name,
            cardinality=cardinality,
            is_seq=True,
            embedding_dim=64,
            feature_hint=FeatureHint.ITEM_ID,
        )
        .build()
    )

    with pytest.raises(ValueError) as exc:
        TorchSequentialValidationDataset._check_if_schema_match(tensor_schema, tensor_schema_gt)

    assert str(exc.value) == exception_msg


@pytest.mark.torch
def test_validation_dataset(sequential_dataset, item_user_sequential_dataset):
    df = TorchSequentialValidationDataset(
        sequential_dataset,
        sequential_dataset,
        item_user_sequential_dataset,
        max_sequence_length=5,
    )

    assert len(df) == 4
    assert df[0].query_id == 0


@pytest.mark.torch
@pytest.mark.parametrize(
    "sequence, answer",
    [
        ([0, 1], [-1, 0, 1]),
        ([[0, 1, 3], [4, 5, 6]], [[-1, -1, -1], [0, 1, 3], [4, 5, 6]])
    ],
)
def test_pad_sequence(sequential_dataset, sequence, answer):
    dataset = TorchSequentialDataset(
        sequential_dataset,
        max_sequence_length=3,
        sliding_window_step=2,
        padding_value=-1,
    )

    padded_sequence = dataset._pad_sequence(torch.tensor(sequence, dtype=torch.long)).tolist()
    assert padded_sequence == answer


@pytest.mark.torch
def test_pad_sequence_raise(sequential_dataset):
    dataset = TorchSequentialDataset(
        sequential_dataset,
        max_sequence_length=3,
        sliding_window_step=2,
        padding_value=-1,
    )
    sequence = [[[1, 1]], [[2, 2]]]
    with pytest.raises(ValueError, match="Unsupported shape for sequence"):
        dataset._pad_sequence(torch.tensor(sequence, dtype=torch.long)).tolist()


def _compare_sequence(dataset: TorchSequentialDataset, index: int, feature_name: str, expected: List[int]) -> None:
    query_id, padding_mask, feature_sequence = dataset[index]

    actual_sequence_np = feature_sequence[feature_name].numpy()
    actual_padding_mask_np = padding_mask.numpy()

    expected_sequence_np = np.array(expected)
    expected_padding_mask = expected_sequence_np >= 0

    assert len(query_id) == 1
    assert (actual_padding_mask_np == expected_padding_mask).all()
    assert (actual_sequence_np == expected_sequence_np).all()


def _compare_query_feature(dataset: TorchSequentialDataset, index: int, feature_name: str, expected: int) -> None:
    feature_tensor = dataset[index][-1][feature_name]
    assert len(feature_tensor) == 1
    assert feature_tensor.item() == expected
