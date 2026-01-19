import copy

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch

pytest.importorskip("torch")

from replay.data import FeatureHint, FeatureSource, FeatureType
from replay.data.nn import (
    ParquetModule,
    TensorFeatureInfo,
    TensorFeatureSource,
    TensorSchema,
)
from replay.nn.sequential.sasrec import SasRec
from replay.nn.transforms import (
    CopyTransform,
    GroupTransform,
    NextTokenTransform,
    RenameTransform,
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
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "item_id")],
                feature_hint=FeatureHint.ITEM_ID,
            ),
            TensorFeatureInfo(
                name="cat_list_feature",
                is_seq=True,
                cardinality=5,
                padding_value=4,
                embedding_dim=64,
                feature_type=FeatureType.CATEGORICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "cat_list_feature")],
            ),
            TensorFeatureInfo(
                name="num_feature",
                is_seq=True,
                tensor_dim=1,
                padding_value=0,
                embedding_dim=64,
                feature_type=FeatureType.NUMERICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "num_feature")],
            ),
            TensorFeatureInfo(
                name="num_list_feature",
                is_seq=True,
                padding_value=0,
                tensor_dim=6,
                embedding_dim=64,
                feature_type=FeatureType.NUMERICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "num_list_feature")],
            ),
            TensorFeatureInfo(
                name="emb_list_feature",
                is_seq=True,
                padding_value=0,
                tensor_dim=64,
                embedding_dim=64,
                feature_type=FeatureType.NUMERICAL_LIST,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "num_list_feature")],
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

    return {
        "feature_tensors": {
            "item_id": item_sequences,
            "num_feature": num_feature_sequences,
            "cat_list_feature": cat_list_feature_sequences,
            "num_list_feature": num_list_feature_sequences,
            "emb_list_feature": emb_list_feature_sequences,
        },
        "padding_mask": padding_mask,
    }


def generate_recsys_dataset(
    tensor_schema: TensorSchema,
    n_users: int = 10,
    max_len: int = 20,
    seed: int = 2,
    cat_list_item_size: int = 3,
    num_list_item_size: int = 5,
):
    np.random.seed(seed)

    rows = []
    for i in range(n_users):
        hist_len = np.random.randint(1, max_len + 1, size=None, dtype=int)
        row = {"user_id": i}

        for feature_info in tensor_schema.all_features:
            if not feature_info.is_seq:
                continue

            if feature_info.feature_type == FeatureType.CATEGORICAL:
                row[feature_info.name] = np.random.randint(0, feature_info.cardinality - 2, size=hist_len).tolist()
            elif feature_info.feature_type == FeatureType.CATEGORICAL_LIST:
                row[feature_info.name] = [
                    np.random.randint(0, feature_info.cardinality - 2, size=hist_len) for _ in range(cat_list_item_size)
                ]
            elif feature_info.feature_type == FeatureType.NUMERICAL:
                row[feature_info.name] = np.random.random(size=(hist_len,)).astype(np.float32)
            elif feature_info.feature_type == FeatureType.NUMERICAL_LIST:
                row[feature_info.name] = [
                    np.random.random(size=(feature_info.tensor_dim)).astype(np.float32) for _ in range(hist_len)
                ]

        rows.append(row)

    return pd.DataFrame.from_records(rows)


@pytest.fixture(scope="module")
def max_len():
    return 7


@pytest.fixture(scope="module")
def seed():
    return 1


@pytest.fixture(scope="module")
def parquet_module_path(tmp_path_factory, tensor_schema, seed, max_len):
    tmp_dir = tmp_path_factory.mktemp("parquet_module")
    path = tmp_dir / f"tmp_{seed}.parquet"

    df = generate_recsys_dataset(tensor_schema, n_users=50, max_len=max_len, seed=seed)

    schema = pa.schema(
        [
            ("item_id", pa.list_(pa.int64())),
            ("user_id", pa.int64()),
            ("cat_list_feature", pa.list_(pa.list_(pa.int64()))),
            ("num_feature", pa.list_(pa.float32())),
            ("num_list_feature", pa.list_(pa.list_(pa.float32()))),
            ("emb_list_feature", pa.list_(pa.list_(pa.float32()))),
        ]
    )
    df.to_parquet(path, index=False, schema=schema)

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
        "val": [
            RenameTransform({"user_id": "query_id", "item_id_mask": "padding_mask"}),
            CopyTransform({"item_id": "train"}),
            CopyTransform({"item_id": "ground_truth"}),
            GroupTransform({"feature_tensors": tensor_schema.names}),
        ],
        "test": [
            RenameTransform({"user_id": "query_id", "item_id_mask": "padding_mask"}),
            GroupTransform({"feature_tensors": tensor_schema.names}),
        ],
    }
    shared_meta = {
        "user_id": {},
        "item_id": {"shape": 8, "padding": tensor_schema["item_id"].padding_value},
        "cat_list_feature": {"shape": [8, 3], "padding": tensor_schema["cat_list_feature"].padding_value},
        "num_feature": {"shape": 8, "padding": tensor_schema["num_feature"].padding_value},
        "num_list_feature": {"shape": [8, 6], "padding": tensor_schema["num_list_feature"].padding_value},
        "emb_list_feature": {"shape": [8, 64], "padding": tensor_schema["emb_list_feature"].padding_value},
    }

    metadata = {
        "train": copy.deepcopy(shared_meta),
        "val": copy.deepcopy(shared_meta),
        "test": copy.deepcopy(shared_meta),
    }

    parquet_module = ParquetModule(
        metadata=metadata,
        transforms=transforms,
        batch_size=batch_size,
        train_path=parquet_module_path,
        val_path=parquet_module_path,
        test_path=parquet_module_path,
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
def sasrec_model(tensor_schema):
    model = SasRec.build_default(
        schema=tensor_schema, embedding_dim=64, num_heads=1, num_blocks=1, max_sequence_length=7, dropout=0.2
    )
    return model
