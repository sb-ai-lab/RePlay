import numpy as np
import pandas as pd
import polars as pl
import pytest

from replay.data import FeatureHint, FeatureType
from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

    from replay.data.nn import PandasSequentialDataset, PolarsSequentialDataset, TensorFeatureInfo, TensorSchema
    from replay.models.nn.sequential.bert4rec import (
        Bert4RecPredictionDataset,
        Bert4RecTrainingDataset,
        Bert4RecValidationDataset,
    )
    from replay.models.nn.sequential.sasrec import (
        SasRecPredictionDataset,
        SasRecTrainingDataset,
        SasRecValidationDataset,
    )


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
                feature_hint=FeatureHint.ITEM_ID,
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
                feature_hint=FeatureHint.ITEM_ID,
                feature_type=FeatureType.CATEGORICAL,
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
                padding_value=-10,
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
                feature_hint=FeatureHint.ITEM_ID,
                feature_type=FeatureType.CATEGORICAL,
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
                tensor_dim=64,
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
            ),
            TensorFeatureInfo(
                "timestamp",
                cardinality=4,
                is_seq=True,
                embedding_dim=64,
                feature_hint=FeatureHint.TIMESTAMP,
                feature_type=FeatureType.CATEGORICAL,
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
                feature_hint=FeatureHint.ITEM_ID,
                feature_type=FeatureType.CATEGORICAL,
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
def polars_item_user_sequential_dataset():
    sequences = pl.from_records(
        [
            (0, np.array([0, 1, 1, 1, 2]), np.array([[0.0, 0.0], [1.1, 1.1], [1.1, 1.1], [1.1, 1.1], [2.2, 2.2]])),
            (1, np.array([0, 1, 3, 1, 2]), np.array([[0.0, 0.0], [1.1, 1.1], [1.1, 1.1], [1.1, 1.1], [2.2, 2.2]])),
            (2, np.array([0, 2, 3, 1, 2]), np.array([[0.0, 0.0], [1.1, 1.1], [1.1, 1.1], [1.1, 1.1], [2.2, 2.2]])),
            (3, np.array([1, 2, 0, 1, 2]), np.array([[0.0, 0.0], [1.1, 1.1], [1.1, 1.1], [1.1, 1.1], [2.2, 2.2]])),
        ],
        schema=[
            "user_id",
            "item_id",
            "num_feature",
        ],
    )

    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_hint=FeatureHint.ITEM_ID,
                feature_type=FeatureType.CATEGORICAL,
            ),
            TensorFeatureInfo(
                "num_feature",
                tensor_dim=2,
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
            ),
        ]
    )

    sequential_dataset = PolarsSequentialDataset(
        tensor_schema=schema,
        query_id_column="user_id",
        item_id_column="item_id",
        sequences=sequences,
    )

    return sequential_dataset


@pytest.fixture(scope="module")
def item_user_num_sequential_dataset():
    sequences = pd.DataFrame(
        [
            (0, np.array([0, 1, 1, 1, 2]), np.array([0.1, 0.2])),
            (1, np.array([0, 1, 3, 1, 2]), np.array([0.1, 0.2])),
            (2, np.array([0, 2, 3, 1, 2]), np.array([0.1, 0.2])),
            (3, np.array([1, 2, 0, 1, 2]), np.array([0.1, 0.2])),
        ],
        columns=["user_id", "item_id", "num_feature"],
    )

    schema = TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                cardinality=6,
                is_seq=True,
                feature_hint=FeatureHint.ITEM_ID,
                feature_type=FeatureType.CATEGORICAL,
            ),
            TensorFeatureInfo(
                "num_feature",
                tensor_dim=64,
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
def train_loader(item_user_sequential_dataset):
    train = Bert4RecTrainingDataset(item_user_sequential_dataset, 5)
    return torch.utils.data.DataLoader(train)


@pytest.fixture(scope="module")
def val_loader(item_user_sequential_dataset):
    val = Bert4RecValidationDataset(
        item_user_sequential_dataset, item_user_sequential_dataset, item_user_sequential_dataset, max_sequence_length=5
    )
    return torch.utils.data.DataLoader(val)


@pytest.fixture(scope="module")
def pred_loader(item_user_sequential_dataset):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    return torch.utils.data.DataLoader(pred)


@pytest.fixture(scope="module")
def train_sasrec_loader(item_user_sequential_dataset):
    train = SasRecTrainingDataset(item_user_sequential_dataset, 5)
    return torch.utils.data.DataLoader(train)


@pytest.fixture(scope="module")
def val_sasrec_loader(item_user_sequential_dataset):
    val = SasRecValidationDataset(
        item_user_sequential_dataset, item_user_sequential_dataset, item_user_sequential_dataset, max_sequence_length=5
    )
    return torch.utils.data.DataLoader(val)


@pytest.fixture(scope="module")
def pred_sasrec_loader(item_user_sequential_dataset):
    pred = SasRecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    return torch.utils.data.DataLoader(pred)
