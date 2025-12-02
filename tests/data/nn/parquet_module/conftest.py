import copy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from replay.data import FeatureHint, FeatureSource, FeatureType
from replay.data.nn import (
    TensorFeatureInfo,
    TensorFeatureSource,
    TensorSchema,
)
from replay.nn.transforms import (
    GroupTransform,
    NextTokenTransform,
    RenameTransform,
)


def generate_recsys_dataset(n_users: int = 10, max_len: int = 20, n_items=30, seed=2):
    np.random.seed(seed)

    rows = []
    for i in range(n_users):
        hist_len = np.random.randint(0, max_len, size=None, dtype=int)

        row = {"user_id": i}
        row["item_id"] = np.random.randint(0, n_items, hist_len).tolist()
        rows.append(row)

    return pd.DataFrame.from_records(rows)


@pytest.fixture
def parquet_module_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("parquet_module")
    path = tmp_dir / "tmp.parquet"

    df = generate_recsys_dataset()
    df.to_parquet(path, index=False)

    return str(path)


@pytest.fixture(scope="module")
def parquet_module_args():
    max_len = 10
    tensor_schema = TensorSchema(
        TensorFeatureInfo(
            name="item_id",
            is_seq=True,
            cardinality=30,
            embedding_dim=64,
            feature_type=FeatureType.CATEGORICAL,
            feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "item_id")],
            feature_hint=FeatureHint.ITEM_ID,
        )
    )
    transforms = {
        "train": [
            NextTokenTransform(label_field="item_id", query_features="user_id", shift=1),
            RenameTransform(
                {"user_id": "query_id", "item_id_mask": "padding_mask", "labels_mask": "labels_padding_mask"}
            ),
            GroupTransform({"features": ["item_id"]}),
        ],
        "val": [
            RenameTransform({"user_id": "query_id", "item_id_mask": "padding_mask"}),
            GroupTransform({"features": ["item_id"]}),
        ],
        "test": [
            RenameTransform({"user_id": "query_id", "item_id_mask": "padding_mask"}),
            GroupTransform({"features": ["item_id"]}),
        ],
    }
    shared_meta = {"user_id": {}, "item_id": {"shape": max_len, "padding": tensor_schema["item_id"].padding_value}}

    metadata = {
        "train": copy.deepcopy(shared_meta),
        "val": copy.deepcopy(shared_meta),
        "test": copy.deepcopy(shared_meta),
    }

    return {"batch_size": 2, "transforms": transforms, "metadata": metadata}
