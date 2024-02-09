import numpy as np
import pandas as pd
import pytest

from replay.data import FeatureHint
from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

    from replay.data.nn import PandasSequentialDataset
    from replay.experimental.nn.data.schema_builder import TensorSchemaBuilder
    from replay.models.nn.sequential.bert4rec import (
        Bert4RecPredictionDataset,
        Bert4RecTrainingDataset,
        Bert4RecValidationDataset,
    )
    from replay.models.nn.sequential.sasrec import SasRecPredictionDataset, SasRecTrainingDataset, SasRecValidationDataset


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


@pytest.fixture(scope="package")
def item_user_num_sequential_dataset():
    sequences = pd.DataFrame(
        [
            (0, np.array([0, 1, 1, 1, 2]), np.array([0.1, 0.2])),
            (1, np.array([0, 1, 3, 1, 2]), np.array([0.1, 0.2])),
            (2, np.array([0, 2, 3, 1, 2]), np.array([0.1, 0.2])),
            (3, np.array([1, 2, 0, 1, 2]), np.array([0.1, 0.2])),
        ],
        columns=[
            "user_id",
            "item_id",
            "num_feature"
        ],
    )

    schema = (
        TensorSchemaBuilder()
        .categorical(
            "item_id",
            cardinality=6,
            is_seq=True,
            feature_hint=FeatureHint.ITEM_ID,
        ).numerical(
            "num_feature",
            tensor_dim=64,
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
