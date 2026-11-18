import pytest

pytest.importorskip("torch")
import torch

from replay.data import FeatureHint, FeatureSource, FeatureType
from replay.data.nn import TensorFeatureInfo, TensorFeatureSource, TensorSchema
from replay.nn.compiled._checkpoint import (
    capture_config,
    config_path_for,
    read_config,
    restore_from_checkpoint,
    save_config,
    schema_from_dict,
    schema_to_dict,
)
from replay.nn.sequential import SasRec

EMB_DIM, CARDINALITY, SEQ_LEN = 32, 50, 16
PARAMS = {
    "embedding_dim": EMB_DIM,
    "num_heads": 2,
    "num_blocks": 1,
    "max_sequence_length": SEQ_LEN,
    "dropout": 0.0,
}


def make_schema():
    return TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                embedding_dim=EMB_DIM,
                padding_value=CARDINALITY,
                cardinality=CARDINALITY,
                feature_hint=FeatureHint.ITEM_ID,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
            ),
            TensorFeatureInfo(
                "price",
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                tensor_dim=1,
                padding_value=0.0,
                embedding_dim=EMB_DIM,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "price")],
            ),
        ]
    )


def _inference(model):
    torch.manual_seed(7)
    feature_tensors = {
        "item_id": torch.randint(0, CARDINALITY, (2, SEQ_LEN)),
        "price": torch.randn(2, SEQ_LEN),
    }
    padding_mask = torch.ones(2, SEQ_LEN, dtype=torch.bool)
    with torch.no_grad():
        return model.forward_inference(feature_tensors, padding_mask)["logits"]


def test_schema_roundtrip():
    schema = make_schema()
    restored = schema_from_dict(schema_to_dict(schema))
    assert restored.names == schema.names
    assert schema_to_dict(restored) == schema_to_dict(schema)


def test_capture_and_read_config(tmp_path):
    schema = make_schema()
    _model, config = capture_config(SasRec, schema, **PARAMS)
    assert config.model_type == "sasrec"
    assert config.from_params == PARAMS

    sidecar = save_config(config, tmp_path / "best.ckpt")
    assert sidecar == tmp_path / "best.ckpt.config.json"
    assert sidecar.is_file()
    assert read_config(tmp_path / "best.ckpt") == config
    assert config_path_for("model.pt").name == "model.pt.config.json"


def test_capture_config_rejects_non_json_kwargs():
    with pytest.raises(ValueError, match="JSON-primitive"):
        capture_config(SasRec, make_schema(), **PARAMS, hidden_dim=object())


def test_restore_from_checkpoint(tmp_path):
    schema = make_schema()
    model, config = capture_config(SasRec, schema, **PARAMS)
    ckpt = tmp_path / "best.pt"
    torch.save(model.state_dict(), ckpt)
    save_config(config, ckpt)

    rebuilt, restored_schema = restore_from_checkpoint(ckpt)
    assert restored_schema.names == schema.names
    assert torch.equal(_inference(rebuilt), _inference(model))


def test_restore_from_lightning_checkpoint(tmp_path):
    schema = make_schema()
    model, config = capture_config(SasRec, schema, **PARAMS)
    ckpt = tmp_path / "lightning.ckpt"
    prefixed = {f"model.{key}": value for key, value in model.state_dict().items()}
    torch.save({"state_dict": prefixed, "epoch": 3}, ckpt)
    save_config(config, ckpt)

    rebuilt, _ = restore_from_checkpoint(ckpt)
    assert torch.equal(_inference(rebuilt), _inference(model))


def test_restore_with_explicit_config_path(tmp_path):
    schema = make_schema()
    model, config = capture_config(SasRec, schema, **PARAMS)
    ckpt = tmp_path / "best.pt"
    torch.save(model.state_dict(), ckpt)
    sidecar = save_config(config, tmp_path / "elsewhere.json")

    rebuilt, _ = restore_from_checkpoint(ckpt, config_path=sidecar)
    assert torch.equal(_inference(rebuilt), _inference(model))


def test_restore_missing_sidecar(tmp_path):
    ckpt = tmp_path / "orphan.ckpt"
    torch.save({}, ckpt)
    with pytest.raises(FileNotFoundError, match="config sidecar"):
        restore_from_checkpoint(ckpt)


def test_restore_unknown_model_type(tmp_path):
    ckpt = tmp_path / "mystery.ckpt"
    torch.save({}, ckpt)
    sidecar = tmp_path / "mystery.ckpt.config.json"
    sidecar.write_text('{"model_type": "als", "from_params": {}, "schema": {"features": []}}')
    with pytest.raises(ValueError, match="No rebuild support"):
        restore_from_checkpoint(ckpt)
