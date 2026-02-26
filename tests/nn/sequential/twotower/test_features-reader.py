import pytest

pytest.importorskip("torch")
from replay.nn.sequential.twotower import FeaturesReader


@pytest.mark.parametrize(
    "metadata_features",
    [
        (["item_id", "num_feature", "cat_list_feature", "num_list_feature"]),
        (["item_id", "num_feature", "cat_list_feature", "num_list_feature", "emb_list_feature", "emb_list_feature_2"]),
    ],
    ids=["missing feature in metadata", "extra feature in metadata"],
)
def test_features_reader_raises_error(tensor_schema, item_features_path, metadata_features):
    metadata = {
        "item_id": {},
        "num_feature": {},
        "cat_list_feature": {"shape": 3, "padding": tensor_schema["cat_list_feature"].padding_value},
        "num_list_feature": {
            "shape": tensor_schema["num_list_feature"].tensor_dim,
            "padding": tensor_schema["num_list_feature"].padding_value,
        },
        "emb_list_feature": {
            "shape": tensor_schema["emb_list_feature"].tensor_dim,
            "padding": tensor_schema["emb_list_feature"].padding_value,
        },
        "emb_list_feature_2": {
            "shape": tensor_schema["emb_list_feature"].tensor_dim,
            "padding": tensor_schema["emb_list_feature"].padding_value,
        },
    }
    with pytest.raises(ValueError):
        FeaturesReader(
            schema=tensor_schema,
            path=item_features_path,
            metadata={k: v for k, v in metadata.items() if k in metadata_features},
        )


def test_features_reader_raises_error_missing_item_id(tensor_schema, item_features_path):
    with pytest.raises(ValueError):
        FeaturesReader(
            schema=tensor_schema.subset(["num_feature"]), path=item_features_path, metadata={"num_feature": {}}
        )
