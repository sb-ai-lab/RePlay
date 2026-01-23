import copy

import pandas as pd
import pytest

pytest.importorskip("torch")

from replay.data import FeatureHint, FeatureSource, FeatureType
from replay.data.nn import (
    TensorFeatureInfo,
    TensorFeatureSource,
    TensorSchema,
)
from replay.nn.transform.template import make_default_sasrec_transforms


@pytest.fixture
def parquet_module_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("parquet_module")
    path = tmp_dir / "tmp.parquet"

    df = pd.DataFrame({"item_id": [0]})
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
    transforms = make_default_sasrec_transforms(tensor_schema, query_column="user_id")

    shared_meta = {"user_id": {}, "item_id": {"shape": max_len, "padding": tensor_schema["item_id"].padding_value}}

    metadata = {
        "train": copy.deepcopy(shared_meta),
        "validate": copy.deepcopy(shared_meta),
        "test": copy.deepcopy(shared_meta),
    }

    return {"batch_size": 2, "transforms": transforms, "metadata": metadata}
