import pytest

pytest.importorskip("torch")
import torch

from replay.nn.compiled._manifest import ExportManifest, FeatureRecord, numpy_dtype_name


def _cat_info():
    from replay.data import FeatureHint, FeatureSource, FeatureType
    from replay.data.nn import TensorFeatureInfo, TensorFeatureSource

    return TensorFeatureInfo(
        "item_id",
        is_seq=True,
        feature_type=FeatureType.CATEGORICAL,
        embedding_dim=8,
        padding_value=15,
        cardinality=15,
        feature_hint=FeatureHint.ITEM_ID,
        feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
    )


def _num_info():
    from replay.data import FeatureSource, FeatureType
    from replay.data.nn import TensorFeatureInfo, TensorFeatureSource

    return TensorFeatureInfo(
        "price",
        is_seq=True,
        feature_type=FeatureType.NUMERICAL,
        tensor_dim=1,
        padding_value=0.0,
        embedding_dim=8,
        feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "price")],
    )


def test_feature_record_from_categorical_info():
    record = FeatureRecord.from_tensor_info(_cat_info())
    assert record.name == "item_id"
    assert record.dtype == "int64"
    assert record.is_categorical
    assert not record.is_list
    assert record.cardinality == 15
    assert record.padding_value == 15
    assert record.tensor_dim is None


def test_feature_record_from_numerical_info():
    record = FeatureRecord.from_tensor_info(_num_info())
    assert record.dtype == "float32"
    assert not record.is_categorical
    assert record.tensor_dim == 1
    assert record.cardinality is None


def test_numpy_dtype_name_unsupported():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        numpy_dtype_name(torch.float16)


def test_manifest_roundtrip():
    manifest = ExportManifest(
        model_type="sasrec",
        exporter="dynamo",
        opset_version=20,
        input_kinds=["feature", "padding_mask"],
        input_names=["item_id", "padding_mask"],
        output_names=["logits"],
        candidates_policy="full_vocab",
        features=[FeatureRecord.from_tensor_info(_cat_info())],
        input_shapes={"item_id": [4, "seq_len"], "padding_mask": [4, "seq_len"]},
        versions={"torch": "2.14.1"},
    )
    assert manifest.feature_names() == ["item_id"]

    restored = ExportManifest.from_dict(manifest.to_dict())
    assert restored == manifest
    assert restored.input_shapes["item_id"] == [4, "seq_len"]
    assert isinstance(restored.features[0], FeatureRecord)
