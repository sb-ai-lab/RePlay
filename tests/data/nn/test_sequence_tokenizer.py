import os
from typing import List

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from replay.data import Dataset, FeatureHint, FeatureSource
from replay.preprocessing import LabelEncoder
from replay.utils import TORCH_AVAILABLE, MissingImportType

if TORCH_AVAILABLE:
    from replay.data.nn import SequenceTokenizer, SequentialDataset, TensorFeatureSource, TensorSchema
    from replay.data.nn.sequence_tokenizer import _SequenceProcessor
    from replay.experimental.nn.data.schema_builder import TensorSchemaBuilder
else:
    TensorSchema = MissingImportType
    SequentialDataset = MissingImportType
    PandasSequentialDataset = MissingImportType


@pytest.mark.torch
def test_item_ids_are_grouped_to_sequences(small_dataset: Dataset, only_item_id_schema: TensorSchema):
    sequential_dataset = SequenceTokenizer(only_item_id_schema).fit_transform(small_dataset)

    _compare_sequence(sequential_dataset, 0, "item_id", [0, 1])
    _compare_sequence(sequential_dataset, 1, "item_id", [0, 2, 3])
    _compare_sequence(sequential_dataset, 2, "item_id", [1])
    _compare_sequence(sequential_dataset, 3, "item_id", [0, 1, 2, 3, 4, 5])


@pytest.mark.torch
def test_item_ids_are_grouped_to_sequences_with_subset(
    small_dataset: Dataset, item_id_and_item_feature_schema: TensorSchema
):
    tokenizer = SequenceTokenizer(item_id_and_item_feature_schema).fit(small_dataset)
    sequential_dataset = tokenizer.transform(small_dataset, tensor_features_to_keep=["item_id"])

    _compare_sequence(sequential_dataset, 0, "item_id", [0, 1])
    _compare_sequence(sequential_dataset, 1, "item_id", [0, 2, 3])
    _compare_sequence(sequential_dataset, 2, "item_id", [1])
    _compare_sequence(sequential_dataset, 3, "item_id", [0, 1, 2, 3, 4, 5])

    for tensor_feature_name in sequential_dataset.schema.keys():
        assert tensor_feature_name in {"item_id"}

    with pytest.raises(KeyError):
        sequential_dataset.get_sequence(0, "some_item_feature")


@pytest.mark.torch
def test_encoding_if_features_missing(small_dataset_no_features: Dataset, only_item_id_schema: TensorSchema):
    sequential_dataset = SequenceTokenizer(only_item_id_schema).fit_transform(small_dataset_no_features)

    assert sequential_dataset.get_query_id(0) == 0
    assert sequential_dataset.get_query_id(1) == 1
    assert sequential_dataset.get_query_id(2) == 2
    assert sequential_dataset.get_query_id(3) == 3

    _compare_sequence(sequential_dataset, 0, "item_id", [0, 1])
    _compare_sequence(sequential_dataset, 1, "item_id", [0, 2, 3])
    _compare_sequence(sequential_dataset, 2, "item_id", [1])
    _compare_sequence(sequential_dataset, 3, "item_id", [0, 1, 2, 3, 4, 5])


@pytest.mark.torch
def test_interactions_features_are_grouped_to_sequences(small_dataset: Dataset):
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id"),
            feature_hint=FeatureHint.ITEM_ID,
        )
        .categorical(
            "timestamp",
            cardinality=12,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "timestamp"),
            feature_hint=FeatureHint.TIMESTAMP,
        )
        .build()
    )
    sequential_dataset = SequenceTokenizer(schema).fit_transform(small_dataset)

    _compare_sequence(sequential_dataset, 0, "timestamp", [0, 1])
    _compare_sequence(sequential_dataset, 1, "timestamp", [2, 3, 4])
    _compare_sequence(sequential_dataset, 2, "timestamp", [5])
    _compare_sequence(sequential_dataset, 3, "timestamp", [6, 7, 8, 9, 10, 11])


@pytest.mark.torch
def test_item_features_are_grouped_to_sequences(small_dataset: Dataset, item_id_and_item_feature_schema: TensorSchema):
    sequential_dataset = SequenceTokenizer(item_id_and_item_feature_schema).fit_transform(small_dataset)

    _compare_sequence(sequential_dataset, 0, "some_item_feature", [0, 1])
    _compare_sequence(sequential_dataset, 1, "some_item_feature", [0, 2, 3])
    _compare_sequence(sequential_dataset, 2, "some_item_feature", [1])
    _compare_sequence(sequential_dataset, 3, "some_item_feature", [0, 1, 2, 3, 4, 5])


@pytest.mark.torch
def test_user_features_are_grouped_to_sequences(small_dataset: Dataset):
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id"),
            feature_hint=FeatureHint.ITEM_ID,
        )
        .categorical(
            "some_user_feature",
            cardinality=2,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.QUERY_FEATURES, "some_user_feature"),
        )
        .build()
    )

    sequential_dataset = SequenceTokenizer(schema).fit_transform(small_dataset)

    _compare_sequence(sequential_dataset, 0, "some_user_feature", [0, 0])
    _compare_sequence(sequential_dataset, 1, "some_user_feature", [1, 1, 1])
    _compare_sequence(sequential_dataset, 2, "some_user_feature", [0])
    _compare_sequence(sequential_dataset, 3, "some_user_feature", [0, 0, 0, 0, 0, 0])


@pytest.mark.torch
def test_user_features_handled_as_scalars(small_dataset: Dataset):
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id"),
            feature_hint=FeatureHint.ITEM_ID,
        )
        .categorical(
            "some_user_feature",
            cardinality=2,
            is_seq=False,
            feature_source=TensorFeatureSource(FeatureSource.QUERY_FEATURES, "some_user_feature"),
        )
        .build()
    )

    sequential_dataset = SequenceTokenizer(schema).fit_transform(small_dataset)

    _compare_sequence(sequential_dataset, 0, "some_user_feature", [0])
    _compare_sequence(sequential_dataset, 1, "some_user_feature", [1])
    _compare_sequence(sequential_dataset, 2, "some_user_feature", [0])
    _compare_sequence(sequential_dataset, 3, "some_user_feature", [0])


@pytest.mark.torch
def test_tokenizer_properties(item_id_and_item_feature_schema, small_dataset):
    tokenizer = SequenceTokenizer(item_id_and_item_feature_schema).fit(small_dataset)

    assert isinstance(tokenizer.tensor_schema, TensorSchema)
    assert isinstance(tokenizer.query_id_encoder, LabelEncoder)
    assert isinstance(tokenizer.item_id_encoder, LabelEncoder)
    assert isinstance(tokenizer.query_and_item_id_encoder, LabelEncoder)
    assert isinstance(tokenizer.interactions_encoder, LabelEncoder)
    assert isinstance(tokenizer.query_features_encoder, LabelEncoder)
    assert isinstance(tokenizer.item_features_encoder, LabelEncoder)


@pytest.mark.torch
def test_no_timestamp_dataset(only_item_id_schema, small_dataset_no_timestamp):
    SequenceTokenizer(only_item_id_schema).fit_transform(small_dataset_no_timestamp)


@pytest.mark.torch
def test_invalid_tensor_schema(fake_schema):
    with pytest.raises(ValueError) as exc1:
        SequenceTokenizer(fake_schema.subset(["item_id", "some_user_feature"]))

    with pytest.raises(ValueError) as exc2:
        SequenceTokenizer(fake_schema.subset(["item_id", "some_item_feature"]))

    with pytest.raises(ValueError) as exc3:
        SequenceTokenizer(fake_schema.subset(["item_id", "some_item_feature_2"]))

    with pytest.raises(ValueError):
        SequenceTokenizer(fake_schema.subset(["item_id", "some_item_feature_3"]))

    assert str(exc1.value) == "All tensor features must have sources defined"
    assert str(exc2.value) == "Interaction features must be treated as sequential"
    assert str(exc3.value) == "Item features must be treated as sequential"


@pytest.mark.torch
@pytest.mark.parametrize(
    "subset, exception_msg",
    [
        (["item_id", "some_item_feature_3"], "Found unexpected table '' in tensor schema"),
        (["item_id", "some_user_feature_2"], "Expected column 'some_user_feature_2' in dataset"),
        (["item_id", "some_user_feature", "some_user_feature_2"], "Expected column 'some_user_feature_2' in dataset"),
        (
            ["item_id", "some_user_feature_3"],
            "Expected column 'some_user_feature_3', but query features are not specified",
        ),
        (
            ["item_id", "some_item_feature_2"],
            "Expected column 'some_item_feature_2', but item features are not specified",
        ),
    ],
)
def test_matching_schema_dataset_exceptions(fake_schema, fake_small_dataset, subset, exception_msg):
    with pytest.raises(ValueError) as exc:
        SequenceTokenizer._check_if_tensor_schema_matches_data(
            fake_small_dataset,
            fake_schema.subset(subset),
        )

    assert str(exc.value) == exception_msg


@pytest.mark.torch
@pytest.mark.parametrize(
    "feature_name, source_table, exception_msg",
    [
        (
            "fake_query_feature",
            FeatureSource.QUERY_FEATURES,
            "Expected column 'fake_query_feature' in query features data frame",
        ),
        (
            "fake_item_feature",
            FeatureSource.ITEM_FEATURES,
            "Expected column 'fake_item_feature' in item features data frame",
        ),
    ],
)
def test_invalid_tensor_schema_and_dataset(small_dataset, feature_name, source_table, exception_msg):
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id"),
            feature_hint=FeatureHint.ITEM_ID,
        )
        .categorical(
            feature_name,
            cardinality=2,
            is_seq=True,
            feature_source=TensorFeatureSource(source_table, feature_name),
        )
        .categorical(
            "user_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "user_id"),
            feature_hint=FeatureHint.QUERY_ID,
        )
        .build()
    )

    with pytest.raises(ValueError) as exc:
        SequenceTokenizer._check_if_tensor_schema_matches_data(
            small_dataset,
            schema,
        )

    assert str(exc.value) == exception_msg


@pytest.mark.torch
@pytest.mark.parametrize(
    "user_hint, item_hint, exception_msg",
    [
        (
            FeatureHint.ITEM_ID,
            FeatureHint.QUERY_ID,
            "Tensor schema query ID source colum does not match query ID in data frame",
        ),
        (
            FeatureHint.ITEM_ID,
            None,
            "Tensor schema item ID source colum does not match item ID in data frame",
        ),
    ],
)
def test_items_users_names_mismatch(small_dataset, user_hint, item_hint, exception_msg):
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id"),
            feature_hint=item_hint,
        )
        .categorical(
            "user_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "user_id"),
            feature_hint=user_hint,
        )
        .build()
    )

    with pytest.raises(ValueError) as exc:
        SequenceTokenizer._check_if_tensor_schema_matches_data(
            small_dataset,
            schema,
        )

    assert str(exc.value) == exception_msg


@pytest.mark.torch
def test_item_id_feature_not_specified(small_dataset):
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id"),
        )
        .categorical(
            "user_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "user_id"),
            feature_hint=FeatureHint.QUERY_ID,
        )
        .build()
    )

    with pytest.raises(ValueError) as exc:
        SequenceTokenizer._check_if_tensor_schema_matches_data(
            small_dataset,
            schema,
        )

    assert str(exc.value) == "Tensor schema must have item id feature defined"


@pytest.mark.torch
def test_invalid_source_table():
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id"),
            feature_hint=FeatureHint.ITEM_ID,
        )
        .categorical(
            "feature",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource("", "feature"),
        )
        .build()
    )

    with pytest.raises(AssertionError) as exc:
        SequenceTokenizer._get_features_filter_from_schema(schema, "user_id", "item_id")

    assert str(exc.value) == "Unknown tensor feature source"


@pytest.mark.torch
def test_unknown_source_table_in_processor():
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource("", "item_id"),
            feature_hint=FeatureHint.ITEM_ID,
        )
        .build()
    )

    with pytest.raises(AssertionError) as exc:
        _SequenceProcessor(schema, "user_id", "item_id", pd.DataFrame()).process_feature("item_id")

    assert str(exc.value) == "Unknown tensor feature source table"


@pytest.mark.torch
def test_unknown_feature_type_in_process():
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id"),
            feature_hint=FeatureHint.ITEM_ID,
        )
        .build()
    )
    schema["item_id"]._feature_type = None

    with pytest.raises(AssertionError) as exc:
        _SequenceProcessor(schema, "user_id", "item_id", pd.DataFrame()).process_feature("item_id")

    assert str(exc.value) == "Unknown tensor feature type"


def _compare_sequence(dataset: SequentialDataset, index: int, feature_name: str, expected: List[int]):
    sequence = dataset.get_sequence(index, feature_name)
    assert len(sequence) == len(expected)
    assert (sequence == np.array(expected)).all()


@pytest.mark.torch
def test_save_and_load(small_dataset: Dataset, only_item_id_schema: TensorSchema):
    tokenizer = SequenceTokenizer(only_item_id_schema).fit(small_dataset)
    before_save = tokenizer.transform(small_dataset)

    tokenizer.save("sequence_tokenizer.pth")
    del tokenizer

    tokenizer = SequenceTokenizer.load("sequence_tokenizer.pth")
    after_save = tokenizer.transform(small_dataset)

    _compare_sequence(after_save, 0, "item_id", [0, 1])
    _compare_sequence(after_save, 1, "item_id", [0, 2, 3])
    _compare_sequence(after_save, 2, "item_id", [1])
    _compare_sequence(after_save, 3, "item_id", [0, 1, 2, 3, 4, 5])
    _compare_sequence(before_save, 0, "item_id", after_save.get_sequence(0, "item_id"))
    _compare_sequence(before_save, 1, "item_id", after_save.get_sequence(1, "item_id"))
    _compare_sequence(before_save, 2, "item_id", after_save.get_sequence(2, "item_id"))
    _compare_sequence(before_save, 3, "item_id", after_save.get_sequence(3, "item_id"))


@pytest.mark.torch
def test_save_and_load_different_features_to_keep(
    small_dataset: Dataset, item_id_and_item_feature_schema: TensorSchema
):
    tokenizer = SequenceTokenizer(item_id_and_item_feature_schema).fit(small_dataset)
    item_id_transformed = tokenizer.transform(small_dataset, tensor_features_to_keep=["item_id"])

    tokenizer.save("sequence_tokenizer.pth")
    del tokenizer

    tokenizer = SequenceTokenizer.load("sequence_tokenizer.pth")
    some_item_feature_transformed = tokenizer.transform(small_dataset, tensor_features_to_keep=["some_item_feature"])

    _compare_sequence(item_id_transformed, 0, "item_id", [0, 1])
    _compare_sequence(item_id_transformed, 1, "item_id", [0, 2, 3])
    _compare_sequence(item_id_transformed, 2, "item_id", [1])
    _compare_sequence(item_id_transformed, 3, "item_id", [0, 1, 2, 3, 4, 5])
    _compare_sequence(some_item_feature_transformed, 0, "some_item_feature", [0, 1])
    _compare_sequence(some_item_feature_transformed, 1, "some_item_feature", [0, 2, 3])
    _compare_sequence(some_item_feature_transformed, 2, "some_item_feature", [1])
    _compare_sequence(some_item_feature_transformed, 3, "some_item_feature", [0, 1, 2, 3, 4, 5])
    os.remove("sequence_tokenizer.pth")
