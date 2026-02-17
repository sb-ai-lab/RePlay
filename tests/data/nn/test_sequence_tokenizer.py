import os
import shutil

import pandas as pd
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from replay.data import FeatureHint, FeatureSource, FeatureType
from replay.preprocessing import LabelEncoder
from replay.utils import TORCH_AVAILABLE, MissingImport

if TORCH_AVAILABLE:
    from replay.data.nn import (
        SequenceTokenizer,
        SequentialDataset,
        TensorFeatureInfo,
        TensorFeatureSource,
        TensorSchema,
    )
    from replay.data.nn.sequence_tokenizer import _PandasSequenceProcessor, _PolarsSequenceProcessor
else:
    TensorSchema = MissingImport
    SequentialDataset = MissingImport
    PandasSequentialDataset = MissingImport


def _compare_sequence(
    dataset: SequentialDataset,
    tokenizer: SequenceTokenizer,
    feature_name: str,
    answers: dict,
    feature_inverse_mapping: dict | None = None,
):
    for query in dataset.get_all_query_ids():
        sequence = dataset.get_sequence_by_query_id(query, feature_name).tolist()
        if feature_inverse_mapping:
            if isinstance(sequence[0], list):
                sequence = [[feature_inverse_mapping[y] for y in x] for x in sequence]
            else:
                sequence = [feature_inverse_mapping[x] for x in sequence]
        query = tokenizer.query_id_encoder.inverse_mapping["user_id"][query]
        assert len(sequence) == len(answers[query])
        assert sequence == answers[query]


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
def test_item_ids_are_grouped_to_sequences(dataset, only_item_id_schema: TensorSchema, request):
    data = request.getfixturevalue(dataset)
    tokenizer = SequenceTokenizer(only_item_id_schema)
    sequential_dataset = tokenizer.fit_transform(data)

    answers = {
        1: [1, 2],
        2: [1, 3, 4],
        3: [2],
        4: [1, 2, 3, 4, 5, 6],
    }
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "item_id",
        answers,
        tokenizer.item_id_encoder.inverse_mapping["item_id"],
    )


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
def test_query_features_are_grouped_to_sequences(dataset, request):
    data = request.getfixturevalue(dataset)
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                "timestamp",
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "timestamp")],
                feature_hint=FeatureHint.TIMESTAMP,
            ),
            TensorFeatureInfo(
                "user_cat_seq",
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.QUERY_FEATURES, "user_cat")],
            ),
            TensorFeatureInfo(
                "user_cat_list_seq",
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.QUERY_FEATURES, "user_cat_list")],
            ),
            TensorFeatureInfo(
                "user_cat",
                is_seq=False,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.QUERY_FEATURES, "user_cat")],
            ),
            TensorFeatureInfo(
                "user_cat_list",
                is_seq=False,
                feature_type=FeatureType.CATEGORICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.QUERY_FEATURES, "user_cat_list")],
            ),
        ]
    )
    tokenizer = SequenceTokenizer(schema)
    sequential_dataset = tokenizer.fit_transform(data)

    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "item_id",
        answers={
            1: [1, 2],
            2: [1, 3, 4],
            3: [2],
            4: [1, 2, 3, 4, 5, 6],
        },
        feature_inverse_mapping=tokenizer.item_id_encoder.inverse_mapping["item_id"],
    )
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "user_cat_seq",
        answers={
            1: [1, 1],
            2: [2, 2, 2],
            3: [1],
            4: [1, 1, 1, 1, 1, 1],
        },
        feature_inverse_mapping=tokenizer.query_features_encoder.inverse_mapping["user_cat"],
    )
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "user_cat_list_seq",
        answers={
            1: [[1, 2], [1, 2]],
            2: [[4, 3], [4, 3], [4, 3]],
            3: [[5, 7, 6]],
            4: [[8], [8], [8], [8], [8], [8]],
        },
        feature_inverse_mapping=tokenizer.query_features_encoder.inverse_mapping["user_cat_list"],
    )
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "user_cat",
        answers={
            1: [1],
            2: [2],
            3: [1],
            4: [1],
        },
        feature_inverse_mapping=tokenizer.query_features_encoder.inverse_mapping["user_cat"],
    )
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "user_cat_list",
        answers={
            1: [1, 2],
            2: [4, 3],
            3: [5, 7, 6],
            4: [8],
        },
        feature_inverse_mapping=tokenizer.query_features_encoder.inverse_mapping["user_cat_list"],
    )


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
def test_item_ids_are_grouped_to_sequences_with_subset(
    dataset, item_id_and_item_features_schema: TensorSchema, request
):
    data = request.getfixturevalue(dataset)
    tokenizer = SequenceTokenizer(item_id_and_item_features_schema).fit(data)
    sequential_dataset = tokenizer.transform(data, tensor_features_to_keep=["item_id"])

    answers = {
        1: [1, 2],
        2: [1, 3, 4],
        3: [2],
        4: [1, 2, 3, 4, 5, 6],
    }
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "item_id",
        answers,
        tokenizer.item_id_encoder.inverse_mapping["item_id"],
    )

    for tensor_feature_name in sequential_dataset.schema.keys():
        assert tensor_feature_name in {"item_id"}

    with pytest.raises((KeyError, pl.ColumnNotFoundError)):
        sequential_dataset.get_sequence(0, "some_item_feature")


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset_no_features", "small_dataset_no_features_polars"])
def test_encoding_if_features_missing(dataset, only_item_id_schema: TensorSchema, request):
    data = request.getfixturevalue(dataset)
    tokenizer = SequenceTokenizer(only_item_id_schema)
    sequential_dataset = tokenizer.fit_transform(data)

    assert sequential_dataset.get_query_id(0) == 0
    assert sequential_dataset.get_query_id(1) == 1
    assert sequential_dataset.get_query_id(2) == 2
    assert sequential_dataset.get_query_id(3) == 3

    answers = {
        1: [1, 2],
        2: [1, 3, 4],
        3: [2],
        4: [1, 2, 3, 4, 5, 6],
    }
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "item_id",
        answers,
        tokenizer.item_id_encoder.inverse_mapping["item_id"],
    )


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
def test_interactions_features_are_grouped_to_sequences(dataset, request):
    data = request.getfixturevalue(dataset)
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                feature_type=FeatureType.CATEGORICAL,
                is_seq=True,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                cardinality=6,
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                "timestamp",
                feature_type=FeatureType.NUMERICAL,
                is_seq=True,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "timestamp")],
                feature_hint=FeatureHint.TIMESTAMP,
            ),
        ]
    )

    tokenizer = SequenceTokenizer(schema)
    sequential_dataset = tokenizer.fit_transform(data)

    answers = {
        1: [0, 1],
        2: [2, 3, 4],
        3: [5],
        4: [6, 7, 8, 9, 10, 11],
    }
    _compare_sequence(sequential_dataset, tokenizer, "timestamp", answers)


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
def test_mismatch_of_features_type_raises_error(dataset, request):
    data = request.getfixturevalue(dataset)
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                feature_type=FeatureType.CATEGORICAL,
                is_seq=True,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                cardinality=6,
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                "timestamp",
                feature_type=FeatureType.CATEGORICAL,
                is_seq=True,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "timestamp")],
                feature_hint=FeatureHint.TIMESTAMP,
            ),
        ]
    )

    with pytest.raises(RuntimeError):
        SequenceTokenizer(schema).fit(data)


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
def test_item_features_are_grouped_to_sequences(dataset, item_id_and_item_features_schema: TensorSchema, request):
    data = request.getfixturevalue(dataset)
    tokenizer = SequenceTokenizer(item_id_and_item_features_schema)
    sequential_dataset = tokenizer.fit_transform(data)

    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "item_num",
        answers={
            1: [1.1, 1.2],
            2: [1.1, 1.3, 1.4],
            3: [1.2],
            4: [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        },
    )
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "item_num_list",
        answers={
            1: [[0.5, 2.3], [-1.0, 1.1]],
            2: [[0.5, 2.3], [4.2, 4.8], [-1.1, 0.0]],
            3: [[-1.0, 1.1]],
            4: [[0.5, 2.3], [-1.0, 1.1], [4.2, 4.8], [-1.1, 0.0], [-1.9, 0.12], [2.3, 1.87]],
        },
    )
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "item_cat",
        answers={
            1: [2, 3],
            2: [2, 4, 5],
            3: [3],
            4: [2, 3, 4, 5, 6, 7],
        },
        feature_inverse_mapping=tokenizer.item_features_encoder.inverse_mapping["item_cat"],
    )
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "item_cat_list",
        answers={
            1: [["Animation", "Fantasy"], ["Comedy", "Nature"]],
            2: [["Animation", "Fantasy"], ["Children's", "Action"], ["Nature", "Children's"]],
            3: [["Comedy", "Nature"]],
            4: [
                ["Animation", "Fantasy"],
                ["Comedy", "Nature"],
                ["Children's", "Action"],
                ["Nature", "Children's"],
                ["Animation", "Comedy"],
                ["Fantasy", "Nature"],
            ],
        },
        feature_inverse_mapping=tokenizer.item_features_encoder.inverse_mapping["item_cat_list"],
    )


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
def test_user_features_are_grouped_to_sequences(dataset, request):
    data = request.getfixturevalue(dataset)
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                "user_cat",
                cardinality=2,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.QUERY_FEATURES, "user_cat")],
            ),
        ]
    )

    tokenizer = SequenceTokenizer(schema)
    sequential_dataset = tokenizer.fit_transform(data)

    answers = {
        1: [1, 1],
        2: [2, 2, 2],
        3: [1],
        4: [1, 1, 1, 1, 1, 1],
    }
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "user_cat",
        answers,
        tokenizer.query_features_encoder.inverse_mapping["user_cat"],
    )


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
def test_user_features_handled_as_scalars(dataset, request):
    data = request.getfixturevalue(dataset)
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                "user_cat",
                cardinality=2,
                is_seq=False,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.QUERY_FEATURES, "user_cat")],
            ),
        ]
    )
    tokenizer = SequenceTokenizer(schema)
    sequential_dataset = tokenizer.fit_transform(data)
    answers = {
        1: [1],
        2: [2],
        3: [1],
        4: [1],
    }
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "user_cat",
        answers,
        tokenizer.query_features_encoder.inverse_mapping["user_cat"],
    )


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_numerical_dataset", "small_numerical_dataset_polars"])
def test_process_numerical_features(dataset, request):
    data = request.getfixturevalue(dataset)
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                "feature",
                tensor_dim=1,
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "feature")],
            ),
            TensorFeatureInfo(
                "item_num",
                tensor_dim=1,
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "item_num")],
            ),
            TensorFeatureInfo(
                "user_num",
                tensor_dim=1,
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.QUERY_FEATURES, "user_num")],
            ),
            TensorFeatureInfo(
                "doubled_feature",
                tensor_dim=2,
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "num_feature")],
            ),
            TensorFeatureInfo(
                "num_list_feature",
                is_seq=True,
                feature_type=FeatureType.NUMERICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "num_list_feature")],
            ),
            TensorFeatureInfo(
                "timestamp",
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "timestamp")],
                feature_hint=FeatureHint.TIMESTAMP,
            ),
        ]
    )
    tokenizer = SequenceTokenizer(schema)
    sequential_dataset = tokenizer.fit_transform(data)

    answers = {
        1: [1.1, 1.2],
        2: [1.1, 1.3, 1.4],
        3: [1.2],
        4: [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    }
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "item_num",
        answers,
    )

    for num_feature_name in ["feature", "item_num", "doubled_feature", "user_num"]:
        for query in sequential_dataset.get_all_query_ids():
            query_decoded = tokenizer.query_id_encoder.inverse_mapping["user_id"][query]
            seq = sequential_dataset.get_sequence_by_query_id(query, num_feature_name)
            assert len(seq.shape) == 1
            assert seq.shape[0] == len(answers[query_decoded])

    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "num_list_feature",
        answers={
            1: [[-1.0, 1.0], [-1.1, 1.14]],
            2: [[-1.2, 1.028], [-1.3, 1.34], [-1.4, 1.93]],
            3: [[-1.5, 1.9]],
            4: [[-1.6, 1.8], [-1.7, 1.7], [-1.8, 1.6], [-1.9, 1.5], [-1.95, 1.4], [-1.55, 1.3]],
        },
    )


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_numerical_dataset", "small_numerical_dataset_polars"])
def test_process_categorical_features(dataset, request):
    data = request.getfixturevalue(dataset)
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                "cat_list_feature",
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "cat_list_feature")],
            ),
        ]
    )
    tokenizer = SequenceTokenizer(schema)
    sequential_dataset = tokenizer.fit_transform(data)

    answers = {
        1: [[-1, 0], [1, 2]],
        2: [[3, 4], [12, 11], [9, 10]],
        3: [[8, 7]],
        4: [[0, 5], [6, 7], [7, 13], [-1, 14], [-2, 3], [-3, 9]],
    }
    _compare_sequence(
        sequential_dataset,
        tokenizer,
        "cat_list_feature",
        answers,
        tokenizer.interactions_encoder.inverse_mapping["cat_list_feature"],
    )


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
def test_tokenizer_properties(item_id_and_item_features_schema, dataset, request):
    data = request.getfixturevalue(dataset)
    tokenizer = SequenceTokenizer(item_id_and_item_features_schema).fit(data)
    assert isinstance(tokenizer.tensor_schema, TensorSchema)
    assert isinstance(tokenizer.query_id_encoder, LabelEncoder)
    assert isinstance(tokenizer.item_id_encoder, LabelEncoder)
    assert isinstance(tokenizer.query_and_item_id_encoder, LabelEncoder)
    assert isinstance(tokenizer.query_features_encoder, LabelEncoder)
    assert isinstance(tokenizer.item_features_encoder, LabelEncoder)
    assert tokenizer.interactions_encoder is None


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset_no_timestamp", "small_dataset_no_timestamp_polars"])
def test_no_timestamp_dataset(only_item_id_schema, dataset, request):
    data = request.getfixturevalue(dataset)
    SequenceTokenizer(only_item_id_schema).fit_transform(data)


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
@pytest.mark.parametrize("dataset", ["fake_small_dataset", "fake_small_dataset_polars"])
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
def test_matching_schema_dataset_exceptions(fake_schema, dataset, request, subset, exception_msg):
    data = request.getfixturevalue(dataset)
    with pytest.raises(ValueError) as exc:
        SequenceTokenizer._check_if_tensor_schema_matches_data(
            data,
            fake_schema.subset(subset),
        )

    assert str(exc.value) == exception_msg


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
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
def test_invalid_tensor_schema_and_dataset(dataset, request, feature_name, source_table, exception_msg):
    data = request.getfixturevalue(dataset)
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                feature_name,
                cardinality=2,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(source_table, feature_name)],
            ),
            TensorFeatureInfo(
                "user_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "user_id")],
                feature_hint=FeatureHint.QUERY_ID,
            ),
        ]
    )

    with pytest.raises(ValueError) as exc:
        SequenceTokenizer._check_if_tensor_schema_matches_data(
            data,
            schema,
        )

    assert str(exc.value) == exception_msg


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
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
def test_items_users_names_mismatch(dataset, request, user_hint, item_hint, exception_msg):
    data = request.getfixturevalue(dataset)
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=item_hint,
            ),
            TensorFeatureInfo(
                "user_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "user_id")],
                feature_hint=user_hint,
            ),
        ]
    )

    with pytest.raises(ValueError) as exc:
        SequenceTokenizer._check_if_tensor_schema_matches_data(
            data,
            schema,
        )

    assert str(exc.value) == exception_msg


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
def test_item_id_feature_not_specified(dataset, request):
    data = request.getfixturevalue(dataset)
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
            ),
            TensorFeatureInfo(
                "user_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "user_id")],
                feature_hint=FeatureHint.QUERY_ID,
            ),
        ]
    )

    with pytest.raises(ValueError) as exc:
        SequenceTokenizer._check_if_tensor_schema_matches_data(
            data,
            schema,
        )

    assert str(exc.value) == "Tensor schema must have item id feature defined"


@pytest.mark.torch
def test_invalid_source_table():
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                "feature",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource("", "feature")],
            ),
        ]
    )

    with pytest.raises(AssertionError) as exc:
        SequenceTokenizer._get_features_filter_from_schema(schema, "user_id", "item_id")

    assert str(exc.value) == "Unknown tensor feature source"


@pytest.mark.torch
@pytest.mark.parametrize("dataset", [pd.DataFrame(), pl.DataFrame()])
def test_unknown_source_table_in_cat_processor(dataset):
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource("", "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
        ]
    )
    with pytest.raises(AssertionError) as exc:
        processor_class = _PolarsSequenceProcessor if isinstance(dataset, pl.DataFrame) else _PandasSequenceProcessor
        processor_class(schema, "user_id", "item_id", dataset)._process_feature("item_id")

    assert str(exc.value) == "Unknown tensor feature source table"


@pytest.mark.torch
@pytest.mark.parametrize("dataset_type", [pd.DataFrame, pl.DataFrame])
def test_unknown_source_table_in_num_processor(dataset_type):
    dataset = dataset_type({"item_id": [1, 2, 3], "user_id": [1, 2, 3]})
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "num_feature",
                tensor_dim=6,
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource("", "num_feature")],
            ),
        ]
    )
    with pytest.raises(AssertionError) as exc:
        processor_class = _PolarsSequenceProcessor if isinstance(dataset, pl.DataFrame) else _PandasSequenceProcessor
        processor_class(schema, "user_id", "item_id", dataset)._process_feature("num_feature")

    assert str(exc.value) == "Unknown tensor feature source table"


@pytest.mark.torch
@pytest.mark.parametrize("dataset_type", [pd.DataFrame, pl.DataFrame])
def test_unknown_feature_type_in_process(dataset_type):
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
        ]
    )
    schema["item_id"]._feature_type = None
    dataset = dataset_type({"user_id": [1, 2, 3], "item_id": [1, 2, 3]})
    with pytest.raises(AssertionError) as exc:
        processor_class = _PolarsSequenceProcessor if dataset is pl.DataFrame else _PandasSequenceProcessor
        processor_class(schema, "user_id", "item_id", dataset)._process_feature("item_id")

    assert str(exc.value) == "Unknown tensor feature type"


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
@pytest.mark.parametrize("use_pickle, extension", [(True, "pth"), (False, "replay")])
def test_save_and_load(dataset, request, use_pickle, extension, only_item_id_schema: TensorSchema):
    data = request.getfixturevalue(dataset)
    tokenizer = SequenceTokenizer(only_item_id_schema).fit(data)
    before_save = tokenizer.transform(data)

    tokenizer.save(f"sequence_tokenizer.{extension}", use_pickle=use_pickle)
    del tokenizer

    tokenizer = SequenceTokenizer.load(f"sequence_tokenizer.{extension}", use_pickle=use_pickle)
    after_save = tokenizer.transform(data)

    answers = {
        1: [1, 2],
        2: [1, 3, 4],
        3: [2],
        4: [1, 2, 3, 4, 5, 6],
    }
    _compare_sequence(
        after_save,
        tokenizer,
        "item_id",
        answers,
        tokenizer.item_id_encoder.inverse_mapping["item_id"],
    )
    _compare_sequence(
        before_save,
        tokenizer,
        "item_id",
        answers,
        tokenizer.item_id_encoder.inverse_mapping["item_id"],
    )
    try:
        os.remove(f"sequence_tokenizer.{extension}")
    except IsADirectoryError:
        shutil.rmtree(f"sequence_tokenizer.{extension}")


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
def test_my(item_id_and_timestamp_schema, dataset, request):
    data = request.getfixturevalue(dataset)
    tokenizer = SequenceTokenizer(item_id_and_timestamp_schema)
    sequential_dataset = tokenizer.fit_transform(data)

    assert sequential_dataset.get_sequence_by_query_id(0, "timestamp").ndim == 1
    assert sequential_dataset.get_sequence_by_query_id(1, "timestamp").ndim == 1
    assert sequential_dataset.get_sequence_by_query_id(2, "timestamp").ndim == 1
    assert sequential_dataset.get_sequence_by_query_id(3, "timestamp").ndim == 1


@pytest.mark.torch
@pytest.mark.parametrize("dataset", ["small_dataset", "small_dataset_polars"])
@pytest.mark.parametrize("use_pickle, extension", [(True, "pth"), (False, "replay")])
def test_save_and_load_different_features_to_keep(
    dataset, request, use_pickle, extension, item_id_and_item_features_schema: TensorSchema
):
    data = request.getfixturevalue(dataset)
    tokenizer = SequenceTokenizer(item_id_and_item_features_schema).fit(data)
    item_id_transformed = tokenizer.transform(data, tensor_features_to_keep=["item_id"])

    tokenizer.save(f"sequence_tokenizer.{extension}", use_pickle=use_pickle)
    del tokenizer

    tokenizer = SequenceTokenizer.load(f"sequence_tokenizer.{extension}", use_pickle=use_pickle)
    some_item_feature_transformed = tokenizer.transform(data, tensor_features_to_keep=["item_num"])
    _compare_sequence(
        item_id_transformed,
        tokenizer,
        "item_id",
        answers={
            1: [1, 2],
            2: [1, 3, 4],
            3: [2],
            4: [1, 2, 3, 4, 5, 6],
        },
        feature_inverse_mapping=tokenizer.item_id_encoder.inverse_mapping["item_id"],
    )
    _compare_sequence(
        some_item_feature_transformed,
        tokenizer,
        "item_num",
        answers={
            1: [1.1, 1.2],
            2: [1.1, 1.3, 1.4],
            3: [1.2],
            4: [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        },
    )
    try:
        os.remove(f"sequence_tokenizer.{extension}")
    except IsADirectoryError:
        shutil.rmtree(f"sequence_tokenizer.{extension}")
