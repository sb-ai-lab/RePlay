from typing import List

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("torch")

from replay.data import FeatureHint
from replay.utils import TORCH_AVAILABLE
from replay.utils.common import load_from_replay, save_to_replay

if TORCH_AVAILABLE:
    from replay.data.nn import PandasSequentialDataset, PolarsSequentialDataset
    from replay.experimental.nn.data.schema_builder import TensorSchemaBuilder


@pytest.mark.torch
def test_can_create_sequential_dataset_with_valid_schema(sequential_info, sequential_info_polars):
    PandasSequentialDataset(**sequential_info)
    PolarsSequentialDataset(**sequential_info_polars)


@pytest.mark.torch
@pytest.mark.parametrize("dataset_type", [PandasSequentialDataset, PolarsSequentialDataset])
def test_cannot_create_sequential_dataset_with_invalid_schema(dataset_type, request):
    if dataset_type == PandasSequentialDataset:
        sequential_info = request.getfixturevalue("sequential_info")
    else:
        sequential_info = request.getfixturevalue("sequential_info_polars")
    corrupted_sequences = sequential_info["sequences"].drop(columns=["some_item_feature"])
    sequential_info["sequences"] = corrupted_sequences

    with pytest.raises(ValueError):
        dataset_type(**sequential_info)


@pytest.mark.torch
@pytest.mark.parametrize("dataset_type", [PandasSequentialDataset, PolarsSequentialDataset])
def test_can_get_sequence(dataset_type, request):
    if dataset_type == PandasSequentialDataset:
        sequential_info = request.getfixturevalue("sequential_info")
        sequential_dataset = PandasSequentialDataset(**sequential_info)
    else:
        sequential_info = request.getfixturevalue("sequential_info_polars")
        sequential_dataset = PolarsSequentialDataset(**sequential_info)

    def compare_sequence(index: int, feature_name: str, expected: List[int]):
        assert (sequential_dataset.get_sequence(index, feature_name) == np.array(expected)).all()

    compare_sequence(0, "item_id", [0, 1])
    compare_sequence(1, "item_id", [0, 2, 3])
    compare_sequence(2, "item_id", [1])
    compare_sequence(3, "item_id", [0, 1, 2, 3, 4, 5])

    compare_sequence(0, "some_item_feature", [1, 2])
    compare_sequence(1, "some_item_feature", [1, 3, 4])
    compare_sequence(2, "some_item_feature", [2])
    compare_sequence(3, "some_item_feature", [1, 2, 3, 4, 5, 6])


@pytest.mark.torch
@pytest.mark.parametrize("dataset_type", [PandasSequentialDataset, PolarsSequentialDataset])
def test_can_get_sequence_length(dataset_type, request):
    if dataset_type == PandasSequentialDataset:
        sequential_info = request.getfixturevalue("sequential_info")
        sequential_dataset = PandasSequentialDataset(**sequential_info)
    else:
        sequential_info = request.getfixturevalue("sequential_info_polars")
        sequential_dataset = PolarsSequentialDataset(**sequential_info)

    assert sequential_dataset.get_sequence_length(0) == 2
    assert sequential_dataset.get_sequence_length(1) == 3
    assert sequential_dataset.get_sequence_length(2) == 1
    assert sequential_dataset.get_sequence_length(3) == 6
    assert sequential_dataset.get_max_sequence_length() == 6


@pytest.mark.torch
@pytest.mark.parametrize("dataset_type", [PandasSequentialDataset, PolarsSequentialDataset])
def test_can_get_query_id(dataset_type, request):
    if dataset_type == PandasSequentialDataset:
        sequential_info = request.getfixturevalue("sequential_info")
        sequential_dataset = PandasSequentialDataset(**sequential_info)
    else:
        sequential_info = request.getfixturevalue("sequential_info_polars")
        sequential_dataset = PolarsSequentialDataset(**sequential_info)

    assert sequential_dataset.get_query_id(0) == 0
    assert sequential_dataset.get_query_id(1) == 1
    assert sequential_dataset.get_query_id(2) == 2
    assert sequential_dataset.get_query_id(3) == 3


@pytest.mark.torch
@pytest.mark.parametrize("dataset_type", [PandasSequentialDataset, PolarsSequentialDataset])
def test_intersection_datasets(dataset_type, request):
    if dataset_type == PandasSequentialDataset:
        sequential_info = request.getfixturevalue("sequential_info")
    else:
        sequential_info = request.getfixturevalue("sequential_info_polars")
    dataset = dataset_type(**sequential_info)
    sequences = pd.DataFrame(
        [
            (1, [1], [0, 1], [1, 2]),
            (2, [2], [0, 2, 3], [1, 3, 4]),
            (3, [3], [1], [2]),
            (4, [4], [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]),
        ],
        columns=[
            "user_id",
            "some_user_feature",
            "item_id",
            "some_item_feature",
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
        .categorical(
            "some_user_feature",
            cardinality=4,
            is_seq=False,
        )
        .categorical(
            "some_item_feature",
            cardinality=6,
            is_seq=True,
        )
        .build()
    )

    if dataset_type == PandasSequentialDataset:
        sequential_dataset = PandasSequentialDataset(
            tensor_schema=schema,
            query_id_column="user_id",
            item_id_column="item_id",
            sequences=sequences,
        )
    else:
        sequential_dataset = PolarsSequentialDataset(
            tensor_schema=schema,
            query_id_column="user_id",
            item_id_column="item_id",
            sequences=pl.from_pandas(sequences),
        )

    filtered = dataset_type.keep_common_query_ids(dataset, sequential_dataset)[0]

    assert all(filtered.get_all_query_ids() == [1, 2, 3])


@pytest.mark.core
@pytest.mark.parametrize("dataset_type", [PandasSequentialDataset, PolarsSequentialDataset])
def test_get_sequence_by_query_id(dataset_type, request):
    if dataset_type == PandasSequentialDataset:
        sequential_info = request.getfixturevalue("sequential_info")
    else:
        sequential_info = request.getfixturevalue("sequential_info_polars")
    dataset = dataset_type(**sequential_info)
    assert np.array_equal(dataset.get_sequence_by_query_id(10, "some_item_feature"), np.array([], dtype=np.int64))


@pytest.mark.core
@pytest.mark.parametrize("dataset", ["sequential_info", "sequential_info_polars"])
def test_save_and_load_sequential_dataset(dataset, request, tmp_path):
    data = request.getfixturevalue(dataset)

    if isinstance(data["sequences"], pd.DataFrame):
        obj_type = PandasSequentialDataset
    else:
        obj_type = PolarsSequentialDataset

    sequential_dataset = obj_type(**data)
    save_to_replay(sequential_dataset, tmp_path)
    loaded_sequential_dataset = load_from_replay(tmp_path)

    assert sequential_dataset._query_id_column == loaded_sequential_dataset._query_id_column
    assert sequential_dataset._item_id_column == loaded_sequential_dataset._item_id_column
    assert (
        sequential_dataset._tensor_schema._get_object_args()
        == loaded_sequential_dataset._tensor_schema._get_object_args()
    )
    assert all(sequential_dataset._sequences.columns == loaded_sequential_dataset._sequences.columns)
    assert sequential_dataset._sequences.equals(loaded_sequential_dataset._sequences)
