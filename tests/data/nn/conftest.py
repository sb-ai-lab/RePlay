import numpy as np
import pandas as pd
import polars as pl
import pytest

from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureSource, FeatureType
from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

    from replay.data.nn import PandasSequentialDataset, TensorFeatureInfo, TensorFeatureSource, TensorSchema


@pytest.fixture(scope="module")
def pandas_interactions():
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4],
            "item_id": [1, 2, 1, 3, 4, 2, 1, 2, 3, 4, 5, 6],
            "timestamp": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    )

    return interactions


@pytest.fixture(scope="module")
def query_features():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "user_cat": [1, 2, 1, 1],
            "user_cat_list": [
                [1, 2],
                [4, 3],
                [5, 7, 6],
                [8],
            ],
            "user_num": [1.0, 2.3, 11.8, -1.6],
            "user_num_list": [
                [1.1, 2.1],
                [4.2, 3.2],
                [5.3, 7.3, 6.3],
                [8.4],
            ],
        }
    )


@pytest.fixture(scope="module")
def item_features():
    return pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4, 5, 6],
            "item_cat": [2, 3, 4, 5, 6, 7],
            "item_cat_list": [
                ["Animation", "Fantasy"],
                ["Comedy", "Nature"],
                ["Children's", "Action"],
                ["Nature", "Children's"],
                ["Animation", "Comedy"],
                ["Fantasy", "Nature"],
            ],
            "item_num": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "item_num_list": [
                [0.5, 2.3],
                [-1.0, 1.1],
                [4.2, 4.8],
                [-1.1, 0.0],
                [-1.9, 0.12],
                [2.3, 1.87],
            ],
        }
    )


@pytest.fixture(scope="module")
def fake_schema():
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
                "some_user_feature",
                cardinality=2,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
            ),
            TensorFeatureInfo(
                "some_item_feature",
                cardinality=2,
                is_seq=False,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "some_item_feature")],
            ),
            TensorFeatureInfo(
                "some_item_feature_2",
                cardinality=2,
                is_seq=False,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "some_item_feature_2")],
            ),
            TensorFeatureInfo(
                "some_item_feature_3",
                cardinality=2,
                is_seq=False,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource("", "some_item_feature_3")],
            ),
            TensorFeatureInfo(
                "some_user_feature_2",
                cardinality=2,
                is_seq=False,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "some_user_feature_2")],
            ),
            TensorFeatureInfo(
                "some_user_feature_3",
                cardinality=2,
                is_seq=False,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.QUERY_FEATURES, "some_user_feature_3")],
            ),
            TensorFeatureInfo(
                "some_numerical_feature",
                tensor_dim=2,
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
            ),
        ]
    )

    return schema


@pytest.fixture(scope="module")
def fake_small_feature_schema():
    return FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
            FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP),
        ]
    )


@pytest.fixture(scope="module")
def fake_small_dataset(fake_small_feature_schema, pandas_interactions):
    return Dataset(fake_small_feature_schema, pandas_interactions)


@pytest.fixture(scope="module")
def fake_small_dataset_polars(fake_small_feature_schema, pandas_interactions):
    return Dataset(fake_small_feature_schema, pl.from_pandas(pandas_interactions))


@pytest.fixture(scope="module")
def small_feature_schema():
    return FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
            FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP, FeatureSource.INTERACTIONS),
            FeatureInfo("user_cat", FeatureType.CATEGORICAL, None, FeatureSource.QUERY_FEATURES),
            FeatureInfo("user_cat_list", FeatureType.CATEGORICAL_LIST, None, FeatureSource.QUERY_FEATURES),
            FeatureInfo("user_num", FeatureType.NUMERICAL, None, FeatureSource.QUERY_FEATURES),
            FeatureInfo("user_num_list", FeatureType.NUMERICAL_LIST, None, FeatureSource.QUERY_FEATURES),
            FeatureInfo("item_cat", FeatureType.CATEGORICAL, None, FeatureSource.ITEM_FEATURES),
            FeatureInfo("item_cat_list", FeatureType.CATEGORICAL_LIST, None, FeatureSource.ITEM_FEATURES),
            FeatureInfo("item_num", FeatureType.NUMERICAL, None, FeatureSource.ITEM_FEATURES),
            FeatureInfo("item_num_list", FeatureType.NUMERICAL_LIST, None, FeatureSource.ITEM_FEATURES),
        ]
    )


@pytest.fixture(scope="module")
def small_dataset(small_feature_schema, pandas_interactions, query_features, item_features):
    return Dataset(small_feature_schema, pandas_interactions, query_features, item_features)


@pytest.fixture(scope="module")
def small_dataset_polars(small_feature_schema, pandas_interactions, query_features, item_features):
    return Dataset(
        small_feature_schema,
        pl.from_pandas(pandas_interactions),
        pl.from_pandas(query_features),
        pl.from_pandas(item_features),
    )


@pytest.fixture(scope="module")
def small_numerical_events():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4],
            "item_id": [1, 2, 1, 3, 4, 2, 1, 2, 3, 4, 5, 6],
            "feature": [1, 0, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6],
            "timestamp": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "num_feature": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
            "num_list_feature": [
                [-1.0, 1.0],
                [-1.1, 1.14],
                [-1.2, 1.028],
                [-1.3, 1.34],
                [-1.4, 1.93],
                [-1.5, 1.9],
                [-1.6, 1.8],
                [-1.7, 1.7],
                [-1.8, 1.6],
                [-1.9, 1.5],
                [-1.95, 1.4],
                [-1.55, 1.3],
            ],
            "cat_list_feature": [
                [-1, 0],
                [1, 2],
                [3, 4],
                [12, 11],
                [9, 10],
                [8, 7],
                [0, 5],
                [6, 7],
                [7, 13],
                [-1, 14],
                [-2, 3],
                [-3, 9],
            ],
        }
    )


@pytest.fixture(scope="module")
def small_numerical_feature_schema():
    return FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
            FeatureInfo("feature", FeatureType.NUMERICAL, None, FeatureSource.INTERACTIONS),
            FeatureInfo("timestamp", FeatureType.NUMERICAL, FeatureHint.TIMESTAMP, FeatureSource.INTERACTIONS),
            FeatureInfo("num_feature", FeatureType.NUMERICAL, None, FeatureSource.INTERACTIONS),
            FeatureInfo("num_list_feature", FeatureType.NUMERICAL_LIST, None, FeatureSource.INTERACTIONS),
            FeatureInfo("cat_list_feature", FeatureType.CATEGORICAL_LIST, None, FeatureSource.INTERACTIONS),
            FeatureInfo("item_num", FeatureType.NUMERICAL, None, FeatureSource.ITEM_FEATURES),
        ]
    )


@pytest.fixture(scope="module")
def small_numerical_dataset(small_numerical_feature_schema, small_numerical_events, query_features, item_features):
    return Dataset(
        small_numerical_feature_schema,
        small_numerical_events,
        query_features,
        item_features,
    )


@pytest.fixture(scope="module")
def small_numerical_dataset_polars(
    small_numerical_feature_schema, small_numerical_events, query_features, item_features
):
    return Dataset(
        small_numerical_feature_schema,
        pl.DataFrame(small_numerical_events),
        pl.from_pandas(query_features),
        pl.from_pandas(item_features),
    )


@pytest.fixture(scope="module")
def small_dataset_no_features(fake_small_feature_schema, pandas_interactions):
    return Dataset(
        fake_small_feature_schema,
        pandas_interactions,
    )


@pytest.fixture(scope="module")
def small_dataset_no_features_polars(fake_small_feature_schema, pandas_interactions):
    return Dataset(
        fake_small_feature_schema,
        pl.from_pandas(pandas_interactions),
    )


@pytest.fixture(scope="module")
def small_dataset_no_timestamp_feature_schema():
    return FeatureSchema(
        [
            FeatureInfo("user_id", FeatureType.CATEGORICAL, FeatureHint.QUERY_ID),
            FeatureInfo("item_id", FeatureType.CATEGORICAL, FeatureHint.ITEM_ID),
        ]
    )


@pytest.fixture(scope="module")
def small_dataset_no_timestamp_interactions():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4],
            "item_id": [1, 2, 1, 3, 4, 2, 1, 2, 3, 4, 5, 6],
        }
    )


@pytest.fixture(scope="module")
def small_dataset_no_timestamp(small_dataset_no_timestamp_feature_schema, small_dataset_no_timestamp_interactions):
    return Dataset(small_dataset_no_timestamp_feature_schema, small_dataset_no_timestamp_interactions)


@pytest.fixture(scope="module")
def small_dataset_no_timestamp_polars(
    small_dataset_no_timestamp_feature_schema, small_dataset_no_timestamp_interactions
):
    return Dataset(small_dataset_no_timestamp_feature_schema, pl.from_pandas(small_dataset_no_timestamp_interactions))


@pytest.fixture(scope="module")
def only_item_id_schema():
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            )
        ]
    )
    return schema


@pytest.fixture(scope="module")
def item_id_and_timestamp_schema():
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
                "timestamp",
                is_seq=True,
                tensor_dim=64,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "timestamp")],
                feature_hint=FeatureHint.TIMESTAMP,
            ),
        ]
    )
    return schema


@pytest.fixture(scope="module")
def item_id_and_item_features_schema():
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
                "item_cat",
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "item_cat")],
            ),
            TensorFeatureInfo(
                "item_cat_list",
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "item_cat_list")],
                padding_value=1,
            ),
            TensorFeatureInfo(
                "item_num",
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "item_num")],
            ),
            TensorFeatureInfo(
                "item_num_list",
                is_seq=True,
                feature_type=FeatureType.NUMERICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "item_num_list")],
                padding_value=1,
            ),
        ]
    )
    return schema


@pytest.fixture(scope="class")
def sequential_tensor_schema():
    return TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
            ),
            TensorFeatureInfo(
                "some_user_feature",
                cardinality=4,
                is_seq=False,
                feature_type=FeatureType.CATEGORICAL,
            ),
            TensorFeatureInfo(
                "some_item_feature",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
            ),
            TensorFeatureInfo(
                "some_tensor_feature",
                tensor_dim=2,
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
            ),
        ]
    )


@pytest.fixture(scope="class")
def sequential_df():
    return pd.DataFrame(
        [
            (0, [1], [0, 1], [1, 2], [[1, 2], [1, 2], [1, 2]]),
            (1, [2], [0, 2, 3], [1, 3, 4], [[1, 2], [1, 2]]),
            (2, [3], [1], [2], [[1, 2]]),
            (3, [4], [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [[1, 2], [1, 2], [1, 2], [1, 2]]),
        ],
        columns=[
            "user_id",
            "some_user_feature",
            "item_id",
            "some_item_feature",
            "some_tensor_feature",
        ],
    )


@pytest.fixture(scope="class")
def sequential_info(sequential_tensor_schema, sequential_df):
    return {
        "tensor_schema": sequential_tensor_schema,
        "sequences": sequential_df,
        "query_id_column": "user_id",
        "item_id_column": "item_id",
    }


@pytest.fixture(scope="class")
def sequential_info_polars(sequential_tensor_schema, sequential_df):
    return {
        "tensor_schema": sequential_tensor_schema,
        "sequences": pl.from_pandas(sequential_df),
        "query_id_column": "user_id",
        "item_id_column": "item_id",
    }


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def some_cat_feature():
    return TensorFeatureInfo(
        name="cat_feature",
        feature_type=FeatureType.CATEGORICAL,
        feature_hint=FeatureHint.RATING,
        cardinality=6,
        embedding_dim=6,
        feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id", 0)],
    )


@pytest.fixture(scope="module")
def some_num_tensor_feature(some_num_feature):
    return some_num_feature


@pytest.fixture(scope="module")
def some_cat_tensor_feature(some_cat_feature):
    return some_cat_feature


@pytest.fixture(scope="module")
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

    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                "some_user_feature",
                cardinality=4,
                is_seq=False,
                feature_type=FeatureType.CATEGORICAL,
            ),
            TensorFeatureInfo(
                "some_item_feature",
                tensor_dim=1,
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
            ),
        ]
    )

    sequential_dataset = PandasSequentialDataset(
        tensor_schema=schema,
        query_id_column="user_id",
        item_id_column="item_id",
        sequences=sequences,
    )

    return sequential_dataset


@pytest.fixture(scope="module")
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

    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.ITEM_ID,
                padding_value=-1,
            ),
            TensorFeatureInfo(
                "some_user_feature",
                cardinality=4,
                is_seq=False,
                feature_type=FeatureType.CATEGORICAL,
            ),
            TensorFeatureInfo(
                "some_item_feature",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                padding_value=-2,
            ),
        ]
    )

    sequential_dataset = PandasSequentialDataset(
        tensor_schema=schema,
        query_id_column="user_id",
        item_id_column="item_id",
        sequences=sequences,
    )

    return sequential_dataset


@pytest.fixture(scope="module")
def tensor_schema():
    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=4,
                is_seq=True,
                embedding_dim=64,
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                "some_item_feature",
                cardinality=4,
                is_seq=True,
                embedding_dim=32,
                feature_type=FeatureType.CATEGORICAL,
            ),
            TensorFeatureInfo(
                "some_user_feature",
                cardinality=4,
                is_seq=False,
                embedding_dim=64,
                feature_type=FeatureType.CATEGORICAL,
            ),
            TensorFeatureInfo(
                "some_num_feature",
                is_seq=True,
                tensor_dim=64,
                feature_type=FeatureType.NUMERICAL,
            ),
            TensorFeatureInfo(
                "timestamp",
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                feature_hint=FeatureHint.TIMESTAMP,
            ),
            TensorFeatureInfo(
                "some_cat_feature",
                cardinality=4,
                is_seq=True,
                embedding_dim=64,
                feature_type=FeatureType.CATEGORICAL,
            ),
        ]
    )

    return schema


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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

    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.ITEM_ID,
            ),
        ]
    )

    sequential_dataset = PandasSequentialDataset(
        tensor_schema=schema,
        query_id_column="user_id",
        item_id_column="item_id",
        sequences=sequences,
    )

    return sequential_dataset
