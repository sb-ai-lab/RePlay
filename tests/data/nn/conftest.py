import numpy as np
import pandas as pd
import polars as pl
import pytest

from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureSource, FeatureType
from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

    from replay.data.nn import PandasSequentialDataset, TensorFeatureInfo, TensorFeatureSource
    from replay.experimental.nn.data.schema_builder import TensorSchemaBuilder


@pytest.fixture(scope="package")
def pandas_interactions():
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4],
            "item_id": [1, 2, 1, 3, 4, 2, 1, 2, 3, 4, 5, 6],
            "timestamp": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    )

    return interactions


@pytest.fixture(scope="package")
def query_features():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "some_user_feature": [1, 2, 1, 1],
        }
    )


@pytest.fixture(scope="package")
def item_features():
    return pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4, 5, 6],
            "some_item_feature": [2, 3, 4, 5, 6, 7],
        }
    )


@pytest.fixture
def fake_schema():
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id"),
            feature_hint=FeatureHint.ITEM_ID,
        )
        .categorical("some_user_feature", cardinality=2, is_seq=True, feature_source=None)
        .categorical(
            "some_item_feature",
            cardinality=2,
            is_seq=False,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "some_item_feature"),
        )
        .categorical(
            "some_item_feature_2",
            cardinality=2,
            is_seq=False,
            feature_source=TensorFeatureSource(FeatureSource.ITEM_FEATURES, "some_item_feature_2"),
        )
        .categorical(
            "some_item_feature_3",
            cardinality=2,
            is_seq=False,
            feature_source=TensorFeatureSource("", "some_item_feature_3"),
        )
        .categorical(
            "some_user_feature_2",
            cardinality=2,
            is_seq=False,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "some_user_feature_2"),
        )
        .categorical(
            "some_user_feature_3",
            cardinality=2,
            is_seq=False,
            feature_source=TensorFeatureSource(FeatureSource.QUERY_FEATURES, "some_user_feature_3"),
        )
        .numerical(
            "some_numerical_feature",
            tensor_dim=2,
            is_seq=True,
        )
        .build()
    )

    return schema


@pytest.fixture
def fake_small_dataset(pandas_interactions):
    feature_schema = FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
            FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP),
        ]
    )

    dataset = Dataset(
        feature_schema,
        pandas_interactions,
    )

    return dataset


@pytest.fixture
def fake_small_dataset_polars(pandas_interactions):
    feature_schema = FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
            FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP),
        ]
    )

    dataset = Dataset(
        feature_schema,
        pl.from_pandas(pandas_interactions),
    )

    return dataset


@pytest.fixture
def small_dataset(pandas_interactions, query_features, item_features):
    feature_schema = FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
            FeatureInfo("some_user_feature", FeatureType.CATEGORICAL, None, FeatureSource.QUERY_FEATURES),
            FeatureInfo("some_item_feature", FeatureType.CATEGORICAL, None, FeatureSource.ITEM_FEATURES),
            FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP, FeatureSource.INTERACTIONS),
        ]
    )

    dataset = Dataset(
        feature_schema,
        pandas_interactions,
        query_features,
        item_features,
    )

    return dataset


@pytest.fixture
def small_dataset_polars(pandas_interactions, query_features, item_features):
    feature_schema = FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
            FeatureInfo("some_user_feature", FeatureType.CATEGORICAL, None, FeatureSource.QUERY_FEATURES),
            FeatureInfo("some_item_feature", FeatureType.CATEGORICAL, None, FeatureSource.ITEM_FEATURES),
            FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP, FeatureSource.INTERACTIONS),
        ]
    )

    dataset = Dataset(
        feature_schema,
        pl.from_pandas(pandas_interactions),
        pl.from_pandas(query_features),
        pl.from_pandas(item_features),
    )

    return dataset


@pytest.fixture
def small_numerical_dataset(query_features, item_features):
    feature_schema = FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
            FeatureInfo("feature", FeatureType.NUMERICAL, None, FeatureSource.INTERACTIONS),
            FeatureInfo("some_user_feature", FeatureType.NUMERICAL, None, FeatureSource.QUERY_FEATURES),
            FeatureInfo("some_item_feature", FeatureType.NUMERICAL, None, FeatureSource.ITEM_FEATURES),
            FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP, FeatureSource.INTERACTIONS),
        ]
    )

    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4],
            "item_id": [1, 2, 1, 3, 4, 2, 1, 2, 3, 4, 5, 6],
            "feature": [1, 0, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6],
            "timestamp": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    )

    dataset = Dataset(
        feature_schema,
        interactions,
        query_features,
        item_features,
    )

    return dataset


@pytest.fixture
def small_numerical_dataset_polars(query_features, item_features):
    feature_schema = FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
            FeatureInfo("feature", FeatureType.NUMERICAL, None, FeatureSource.INTERACTIONS),
            FeatureInfo("some_user_feature", FeatureType.NUMERICAL, None, FeatureSource.QUERY_FEATURES),
            FeatureInfo("some_item_feature", FeatureType.NUMERICAL, None, FeatureSource.ITEM_FEATURES),
            FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP, FeatureSource.INTERACTIONS),
        ]
    )

    interactions = pl.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4],
            "item_id": [1, 2, 1, 3, 4, 2, 1, 2, 3, 4, 5, 6],
            "feature": [1, 0, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6],
            "timestamp": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    )

    dataset = Dataset(
        feature_schema,
        interactions,
        pl.from_pandas(query_features),
        pl.from_pandas(item_features),
    )

    return dataset


@pytest.fixture
def small_dataset_no_features(pandas_interactions):
    feature_schema = FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
            FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP, FeatureSource.INTERACTIONS),
        ]
    )

    dataset = Dataset(
        feature_schema,
        pandas_interactions,
    )

    return dataset


@pytest.fixture
def small_dataset_no_features_polars(pandas_interactions):
    feature_schema = FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
            FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP, FeatureSource.INTERACTIONS),
        ]
    )

    dataset = Dataset(
        feature_schema,
        pl.from_pandas(pandas_interactions),
    )

    return dataset


@pytest.fixture
def small_dataset_no_timestamp():
    feature_schema = FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
        ]
    )

    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4],
            "item_id": [1, 2, 1, 3, 4, 2, 1, 2, 3, 4, 5, 6],
        }
    )

    dataset = Dataset(
        feature_schema,
        interactions,
    )

    return dataset


@pytest.fixture
def small_dataset_no_timestamp_polars():
    feature_schema = FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
        ]
    )

    interactions = pl.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4],
            "item_id": [1, 2, 1, 3, 4, 2, 1, 2, 3, 4, 5, 6],
        }
    )

    dataset = Dataset(
        feature_schema,
        interactions,
    )

    return dataset


@pytest.fixture
def only_item_id_schema():
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
    return schema


@pytest.fixture
def item_id_and_timestamp_schema():
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id"),
            feature_hint=FeatureHint.ITEM_ID,
        )
        .numerical(
            "timestamp",
            is_seq=True,
            tensor_dim=64,
            feature_hint=FeatureHint.TIMESTAMP,
            feature_sources=[
                TensorFeatureSource(
                    FeatureSource.INTERACTIONS,
                    "timestamp",
                )
            ],
        )
        .build()
    )
    return schema


@pytest.fixture
def item_id_and_item_feature_schema():
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
            "some_item_feature",
            cardinality=6,
            is_seq=True,
            feature_source=TensorFeatureSource(FeatureSource.ITEM_FEATURES, "some_item_feature"),
        )
        .build()
    )
    return schema


@pytest.fixture
def sequential_info(scope="package"):
    sequences = pd.DataFrame(
        [
            (0, [1], [0, 1], [1, 2]),
            (1, [2], [0, 2, 3], [1, 3, 4]),
            (2, [3], [1], [2]),
            (3, [4], [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]),
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

    return {
        "tensor_schema": schema,
        "sequences": sequences,
        "query_id_column": "user_id",
        "item_id_column": "item_id",
    }


@pytest.fixture()
def sequential_info_polars(sequential_info, scope="package"):
    return {
        "tensor_schema": sequential_info["tensor_schema"],
        "sequences": pl.from_pandas(sequential_info["sequences"]),
        "query_id_column": "user_id",
        "item_id_column": "item_id",
    }


@pytest.fixture()
def some_num_feature():
    return TensorFeatureInfo(
        name="num_feature",
        feature_type=FeatureType.NUMERICAL,
        tensor_dim=1,
        is_seq=True,
        feature_sources=[
            TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id"),
            TensorFeatureSource(FeatureSource.INTERACTIONS, "item_ids"),
        ],
    )


@pytest.fixture()
def some_cat_feature():
    return TensorFeatureInfo(
        name="cat_feature",
        feature_type=FeatureType.CATEGORICAL,
        feature_hint=FeatureHint.RATING,
        cardinality=6,
        embedding_dim=6,
        feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id", 0)],
    )


@pytest.fixture()
def some_num_tensor_feature(some_num_feature):
    return some_num_feature


@pytest.fixture()
def some_cat_tensor_feature(some_cat_feature):
    return some_cat_feature


@pytest.fixture(scope="package")
def wrong_sequential_dataset():
    sequences = pd.DataFrame(
        [
            (0, [1], [0, 1], 1.7373),
            (1, [2], [0, 2, 3], 2.5454),
            (2, [3], [1], 3.6666),
            (3, [4], [0, 1, 2, 3, 4, 5], 4.2121),
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
        .numerical(
            "some_item_feature",
            tensor_dim=1,
            is_seq=True,
        )
        .build()
    )

    sequential_dataset = PandasSequentialDataset(
        tensor_schema=schema,
        query_id_column="user_id",
        item_id_column="item_id",
        sequences=sequences,
    )

    return sequential_dataset


@pytest.fixture(scope="package")
def sequential_dataset():
    sequences = pd.DataFrame(
        [
            (0, [1], [0, 1], [1, 2]),
            (1, [2], [0, 2, 3], [1, 3, 4]),
            (2, [3], [1], [2]),
            (3, [4], [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]),
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

    sequential_dataset = PandasSequentialDataset(
        tensor_schema=schema,
        query_id_column="user_id",
        item_id_column="item_id",
        sequences=sequences,
    )

    return sequential_dataset


@pytest.fixture(scope="package")
def tensor_schema():
    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=4,
            is_seq=True,
            embedding_dim=64,
            feature_hint=FeatureHint.ITEM_ID,
        )
        .categorical(
            "some_item_feature",
            cardinality=4,
            is_seq=True,
            embedding_dim=32,
        )
        .categorical(
            "some_user_feature",
            cardinality=4,
            is_seq=False,
            embedding_dim=64,
        )
        .numerical("some_num_feature", tensor_dim=64, is_seq=True)
        .categorical(
            "timestamp",
            cardinality=4,
            is_seq=True,
            embedding_dim=64,
            feature_hint=FeatureHint.TIMESTAMP,
        )
        .categorical(
            "some_cat_feature",
            cardinality=4,
            is_seq=True,
            embedding_dim=64,
        )
        .build()
    )

    return schema


@pytest.fixture(scope="package")
def simple_masks():
    item_sequences = torch.tensor(
        [
            [0, 0, 0, 1, 2],
            [0, 0, 3, 1, 2],
            [0, 0, 3, 1, 2],
            [0, 0, 0, 1, 2],
        ],
        dtype=torch.long,
    )

    padding_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )

    tokens_mask = torch.tensor(
        [
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=torch.bool,
    )

    timestamp_sequences = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4],
            [0, 0, 1, 2, 3],
            [0, 0, 0, 1, 2],
        ],
        dtype=torch.long,
    )

    return item_sequences, padding_mask, tokens_mask, timestamp_sequences


@pytest.fixture(scope="package")
def item_user_sequential_dataset():
    sequences = pd.DataFrame(
        [
            (0, np.array([0, 1, 1, 1, 2])),
            (1, np.array([0, 1, 3, 1, 2])),
            (2, np.array([0, 2, 3, 1, 2])),
            (3, np.array([1, 2, 0, 1, 2])),
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

    sequential_dataset = PandasSequentialDataset(
        tensor_schema=schema,
        query_id_column="user_id",
        item_id_column="item_id",
        sequences=sequences,
    )

    return sequential_dataset
