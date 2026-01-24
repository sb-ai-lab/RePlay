import copy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch

from replay.data import FeatureHint, FeatureSource, FeatureType
from replay.data.nn import (
    ParquetModule,
    TensorFeatureInfo,
    TensorFeatureSource,
    TensorSchema,
)
from replay.nn.sequential import SasRec
from replay.nn.transform import (
    CopyTransform,
    GroupTransform,
    NextTokenTransform,
    RenameTransform,
    TrimTransform,
    UniformNegativeSamplingTransform,
    UnsqueezeTransform,
)


@pytest.fixture(scope="module")
def tensor_schema():
    tensor_schema = TensorSchema(
        [
            TensorFeatureInfo(
                name="item_id",
                is_seq=True,
                cardinality=41,
                padding_value=40,
                embedding_dim=64,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                name="cat_list_feature",
                is_seq=True,
                cardinality=5,
                padding_value=4,
                embedding_dim=65,
                feature_type=FeatureType.CATEGORICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "cat_list_feature")],
            ),
            TensorFeatureInfo(
                name="num_feature",
                is_seq=True,
                tensor_dim=1,
                padding_value=0,
                embedding_dim=66,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "num_feature")],
            ),
            TensorFeatureInfo(
                name="num_list_feature",
                is_seq=True,
                padding_value=0,
                tensor_dim=6,
                embedding_dim=67,
                feature_type=FeatureType.NUMERICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "num_list_feature")],
            ),
            TensorFeatureInfo(
                name="emb_list_feature",
                is_seq=True,
                padding_value=0,
                tensor_dim=64,
                embedding_dim=68,
                feature_type=FeatureType.NUMERICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "emb_list_feature")],
            ),
        ]
    )
    return tensor_schema


@pytest.fixture(scope="module")
def tensor_schema_with_equal_embedding_dims():
    tensor_schema = TensorSchema(
        [
            TensorFeatureInfo(
                name="item_id",
                is_seq=True,
                cardinality=41,
                padding_value=40,
                embedding_dim=60,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                name="cat_list_feature",
                is_seq=True,
                cardinality=5,
                padding_value=4,
                embedding_dim=60,
                feature_type=FeatureType.CATEGORICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "cat_list_feature")],
            ),
            TensorFeatureInfo(
                name="num_feature",
                is_seq=True,
                tensor_dim=1,
                padding_value=0,
                embedding_dim=60,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "num_feature")],
            ),
            TensorFeatureInfo(
                name="num_list_feature",
                is_seq=True,
                padding_value=0,
                tensor_dim=6,
                embedding_dim=60,
                feature_type=FeatureType.NUMERICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "num_list_feature")],
            ),
            TensorFeatureInfo(
                name="emb_list_feature",
                is_seq=True,
                padding_value=0,
                tensor_dim=64,
                embedding_dim=60,
                feature_type=FeatureType.NUMERICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "emb_list_feature")],
            ),
        ]
    )
    return tensor_schema


@pytest.fixture(scope="module")
def simple_batch():
    item_sequences = torch.LongTensor(
        [
            [40, 40, 40, 1, 2],
            [40, 0, 0, 1, 2],
            [2, 0, 2, 1, 2],
            [40, 40, 2, 1, 0],
        ],
    )
    cat_list_feature_sequences = torch.LongTensor(
        [
            [[4, 4, 4], [4, 4, 4], [4, 4, 4], [3, 2, 1], [2, 4, 3]],
            [[4, 4, 4], [4, 0, 3], [4, 0, 3], [3, 2, 1], [2, 4, 3]],
            [[2, 4, 3], [4, 0, 3], [2, 4, 3], [3, 2, 1], [2, 4, 3]],
            [[4, 4, 4], [4, 4, 4], [2, 4, 3], [3, 2, 1], [4, 0, 3]],
        ]
    )
    num_feature_sequences = torch.FloatTensor(
        [[0.0, 0.0, 0.0, 1.0, 2.0], [0, 0.0, 1.0, 1.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 2.0, 2.0, 2.0]]
    )
    num_list_feature_sequences = torch.rand(4, 5, 6)
    emb_list_feature_sequences = torch.rand(4, 5, 64)

    padding_mask = torch.BoolTensor(
        [
            [0, 0, 0, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
    )
    query_id = torch.LongTensor([0, 1, 2, 3])

    return {
        "feature_tensors": {
            "item_id": item_sequences,
            "num_feature": num_feature_sequences,
            "cat_list_feature": cat_list_feature_sequences,
            "num_list_feature": num_list_feature_sequences,
            "emb_list_feature": emb_list_feature_sequences,
        },
        "padding_mask": padding_mask,
        "query_id": query_id,
    }


def generate_recsys_dataset(
    tensor_schema: TensorSchema,
    n_users: int = 10,
    max_len: int = 20,
    seed: int = 2,
    cat_list_item_size: int = 3,
):
    np.random.seed(seed)

    features_dict = {name: {} for name in tensor_schema.names}

    for feature_info in tensor_schema.all_features:
        if not feature_info.is_seq or feature_info.name == "item_id":
            continue

        d = {}
        for i in range(tensor_schema["item_id"].cardinality - 1):
            if feature_info.feature_type == FeatureType.CATEGORICAL:
                d[i] = np.random.randint(0, feature_info.cardinality - 1, size=None)

            elif feature_info.feature_type == FeatureType.CATEGORICAL_LIST:
                d[i] = np.random.randint(0, feature_info.cardinality - 1, size=cat_list_item_size)

            elif feature_info.feature_type == FeatureType.NUMERICAL:
                d[i] = np.random.random(size=None)

            elif feature_info.feature_type == FeatureType.NUMERICAL_LIST:
                d[i] = np.random.random(size=(feature_info.tensor_dim)).astype(np.float32)

        features_dict[feature_info.name] = copy.deepcopy(d)

    rows = []
    for i in range(n_users):
        hist_len = np.random.randint(1, max_len + 1, size=None, dtype=int)
        row = {"user_id": i}

        row["item_id"] = np.random.randint(0, tensor_schema["item_id"].cardinality - 2, size=hist_len).tolist()

        for feature_info in tensor_schema.all_features:
            if not feature_info.is_seq or feature_info.name == "item_id":
                continue

            sequence = [features_dict[feature_info.name][i] for i in row["item_id"]]

            if feature_info.feature_type == FeatureType.NUMERICAL:
                row[feature_info.name] = np.array(sequence).astype(np.float32)
            else:
                row[feature_info.name] = sequence

        rows.append(row)

    events_df = pd.DataFrame.from_records(rows)
    item_features_df = pd.DataFrame.from_records(features_dict)
    item_features_df["item_id"] = item_features_df.index

    return events_df, item_features_df


@pytest.fixture(scope="module")
def generated_dfs(tensor_schema, seed, max_len):
    return generate_recsys_dataset(tensor_schema, n_users=50, max_len=max_len, seed=seed)


@pytest.fixture(scope="module")
def max_len():
    return 7


@pytest.fixture(scope="module")
def seed():
    return 1


@pytest.fixture(scope="module")
def item_features_path(tmp_path_factory, generated_dfs):
    tmp_dir = tmp_path_factory.mktemp("parquet_module")
    path = tmp_dir / f"item_features_tmp_{seed}.parquet"

    _, item_features = generated_dfs
    item_features.to_parquet(path, index=False)

    return str(path)


@pytest.fixture(scope="module")
def parquet_module_path(tmp_path_factory, generated_dfs):
    tmp_dir = tmp_path_factory.mktemp("parquet_module")
    path = tmp_dir / f"tmp_{seed}.parquet"

    df, _ = generated_dfs
    df.to_parquet(path, index=False)

    return str(path)


@pytest.fixture(scope="module")
def parquet_module(parquet_module_path, tensor_schema, max_len, batch_size=4):
    transforms = {
        "train": [
            NextTokenTransform(
                label_field="item_id", query_features="user_id", shift=1, out_feature_name="positive_labels"
            ),
            RenameTransform(
                {"user_id": "query_id", "item_id_mask": "padding_mask", "positive_labels_mask": "target_padding_mask"}
            ),
            UniformNegativeSamplingTransform(
                vocab_size=tensor_schema["item_id"].cardinality - 2, num_negative_samples=10
            ),
            UnsqueezeTransform("target_padding_mask", -1),
            UnsqueezeTransform("positive_labels", -1),
            GroupTransform({"feature_tensors": tensor_schema.names}),
        ],
        "validate": [
            RenameTransform({"user_id": "query_id", "item_id_mask": "padding_mask"}),
            CopyTransform({"item_id": "train"}),
            CopyTransform({"item_id": "ground_truth"}),
            CopyTransform({"item_id": "seen_ids"}),
            GroupTransform({"feature_tensors": tensor_schema.names}),
        ],
        "test": [
            RenameTransform({"user_id": "query_id", "item_id_mask": "padding_mask"}),
            CopyTransform({"item_id": "train"}),
            CopyTransform({"item_id": "ground_truth"}),
            CopyTransform({"item_id": "seen_ids"}),
            GroupTransform({"feature_tensors": tensor_schema.names}),
        ],
        "predict": [
            RenameTransform({"user_id": "query_id", "item_id_mask": "padding_mask"}),
            CopyTransform({"item_id": "seen_ids"}),
            TrimTransform(max_len, ["seen_ids"]),
            GroupTransform({"feature_tensors": tensor_schema.names}),
        ],
    }

    def create_meta(shape):
        shared_meta = {
            "user_id": {},
            "item_id": {"shape": shape, "padding": tensor_schema["item_id"].padding_value},
            "cat_list_feature": {"shape": [shape, 3], "padding": tensor_schema["cat_list_feature"].padding_value},
            "num_feature": {"shape": shape, "padding": tensor_schema["num_feature"].padding_value},
            "num_list_feature": {"shape": [shape, 6], "padding": tensor_schema["num_list_feature"].padding_value},
            "emb_list_feature": {"shape": [shape, 64], "padding": tensor_schema["emb_list_feature"].padding_value},
        }
        return shared_meta

    metadata = {
        "train": create_meta(shape=max_len + 1),
        "validate": create_meta(shape=max_len),
        "test": create_meta(shape=max_len),
        "predict": create_meta(shape=max_len),
    }

    parquet_module = ParquetModule(
        metadata=metadata,
        transforms=transforms,
        batch_size=batch_size,
        train_path=parquet_module_path,
        validate_path=parquet_module_path,
        test_path=parquet_module_path,
        predict_path=parquet_module_path,
    )
    return parquet_module


@pytest.fixture(scope="module")
def sequential_sample(parquet_module):
    parquet_module.setup("train")
    return parquet_module.compiled_transforms["train"](next(iter(parquet_module.train_dataloader())))


@pytest.fixture(scope="module")
def wrong_sequential_sample(request, sequential_sample):
    sample = copy.deepcopy(sequential_sample)

    defect_type = request.param
    if defect_type == "missing field":
        sample.pop("padding_mask")
    elif defect_type == "wrong length":
        sample["feature_tensors"]["item_id"] = sample["feature_tensors"]["item_id"][:, 1:]
    elif defect_type == "index out of embedding":
        sample["feature_tensors"]["item_id"][0][-1] = 4
    else:
        raise ValueError(defect_type)
    return sample


@pytest.fixture
def sasrec_model(tensor_schema_with_equal_embedding_dims):
    model = SasRec.from_params(
        schema=tensor_schema_with_equal_embedding_dims,
        embedding_dim=60,
        num_heads=1,
        num_blocks=1,
        max_sequence_length=7,
        dropout=0.2,
    )
    return model
