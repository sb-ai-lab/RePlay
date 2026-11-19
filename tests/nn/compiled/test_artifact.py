import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

    from replay.data import FeatureHint, FeatureSource, FeatureType
    from replay.data.nn import TensorFeatureInfo, TensorFeatureSource, TensorSchema
    from replay.nn.compiled import (
        ItemEmbeddings,
        compile,
        export_item_embeddings,
        load,
        load_item_embeddings,
    )

REPO_ROOT = Path(__file__).resolve().parents[3]

FEATURE = {
    "name": "item_id",
    "dtype": "int64",
    "is_categorical": True,
    "is_list": False,
    "is_seq": True,
    "cardinality": 5,
    "tensor_dim": None,
    "embedding_dim": 4,
    "padding_value": 5,
}


def manifest_dict(**overrides):
    data = {
        "model_type": "sasrec",
        "exporter": "dynamo",
        "opset_version": 20,
        "input_kinds": ["feature", "padding_mask"],
        "input_names": ["item_id", "padding_mask"],
        "output_names": ["logits"],
        "candidates_policy": "full_vocab",
        "features": [dict(FEATURE)],
        "input_shapes": {"item_id": ["batch", "seq_len"], "padding_mask": ["batch", "seq_len"]},
        "versions": {},
        "max_seq_len": 4,
        "format_version": 1,
        "encoding": {"padding_side": "front", "keep": "last", "default_value": "padding_value", "mask_true": "valid"},
    }
    data.update(overrides)
    return data


def test_lazy_package_exports():
    import replay.nn.compiled as pkg

    assert callable(pkg.load)
    assert callable(pkg.compile)
    assert "encode_inputs" in dir(pkg)
    with pytest.raises(AttributeError, match="no attribute"):
        pkg.definitely_not_a_symbol


def test_manifest_format_version_gate():
    from replay.nn.compiled._manifest import MANIFEST_FORMAT_VERSION, ExportManifest

    manifest = ExportManifest.from_dict(manifest_dict())
    assert manifest.format_version == MANIFEST_FORMAT_VERSION
    assert manifest.item_matrix is None
    assert ExportManifest.from_dict(manifest.to_dict()) == manifest

    with pytest.raises(ValueError, match="newer than supported"):
        ExportManifest.from_dict(manifest_dict(format_version=MANIFEST_FORMAT_VERSION + 1))

    legacy = manifest_dict()
    del legacy["format_version"]
    with pytest.raises(ValueError, match="predates the artifact format"):
        ExportManifest.from_dict(legacy)


def test_load_requires_openvino(monkeypatch, tmp_path):
    from replay.nn.compiled import _core

    real_find_spec = _core.find_spec
    monkeypatch.setattr(_core, "find_spec", lambda name: None if name == "openvino" else real_find_spec(name))
    with pytest.raises(ImportError, match="openvino"):
        _core.load(tmp_path)


def test_torch_free_serving_imports():
    code = "\n".join(
        [
            "import sys",
            "from replay.nn.compiled import (",
            "    ItemEmbeddings, CompiledModel, compile, encode_inputs, load, load_item_embeddings,",
            ")",
            "assert 'torch' not in sys.modules, 'serving imports pulled torch'",
            "inputs = encode_inputs("
            f"    {manifest_dict()!r}, [[{{'item_id': 'a'}}]], {{'item_id': {{'a': 1}}}}, seq_len=4"
            ")",
            "assert inputs['item_id'].tolist() == [[5, 5, 5, 1]]",
            "import numpy as np, tempfile, os",
            "with tempfile.TemporaryDirectory() as tmp:",
            "    ItemEmbeddings(np.zeros((3, 4), dtype=np.float32), ['item_id']).save(tmp)",
            "    loaded = load_item_embeddings(tmp)",
            "    assert loaded.matrix.shape == (3, 4)",
            "print('TORCH_FREE_OK')",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "TORCH_FREE_OK" in result.stdout


def test_save_rejects_matrix_for_sasrec(tmp_path):
    from replay.nn.compiled import CompiledModel
    from replay.nn.compiled._manifest import ExportManifest, FeatureRecord

    manifest = ExportManifest(
        model_type="sasrec",
        exporter="dynamo",
        opset_version=20,
        input_kinds=["feature", "padding_mask"],
        input_names=["item_id", "padding_mask"],
        output_names=["logits"],
        candidates_policy="full_vocab",
        features=[FeatureRecord(**dict(FEATURE))],
    )
    model = CompiledModel(compiled_model=None, manifest=manifest, onnx_bytes=b"onnx-bytes")
    embeddings = ItemEmbeddings(np.zeros((3, 4), dtype=np.float32), ["item_id"])
    with pytest.raises(ValueError, match="twotower_query"):
        model.save(tmp_path, item_embeddings=embeddings)

    model.save(tmp_path)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["format_version"] == 1
    assert saved["item_matrix"] is None


@pytest.mark.conditional
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch required")
class TestTwoTowerArtifactPair:
    SEQ_LEN, CARDINALITY, EMB_DIM = 16, 50, 32

    class _DictFeaturesReader:
        def __init__(self, features):
            self._features = features
            self.feature_names = list(features)

        def __getitem__(self, name):
            return self._features[name]

    def _schema(self):
        return TensorSchema(
            [
                TensorFeatureInfo(
                    "item_id",
                    is_seq=True,
                    feature_type=FeatureType.CATEGORICAL,
                    embedding_dim=self.EMB_DIM,
                    padding_value=self.CARDINALITY,
                    cardinality=self.CARDINALITY,
                    feature_hint=FeatureHint.ITEM_ID,
                    feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
                ),
            ]
        )

    def _model(self):
        from replay.nn.sequential.twotower.model import TwoTower

        torch.manual_seed(2026)
        model = TwoTower.from_params(
            schema=self._schema(),
            item_features_reader=self._DictFeaturesReader({"item_id": torch.arange(self.CARDINALITY)}),
            embedding_dim=self.EMB_DIM,
            num_heads=2,
            num_blocks=1,
            max_sequence_length=self.SEQ_LEN,
            dropout=0.0,
        )
        model.eval()
        return model

    def _inputs(self):
        rng = np.random.RandomState(2026)
        return {
            "item_id": rng.randint(0, self.CARDINALITY, size=(4, self.SEQ_LEN)).astype(np.int64),
            "padding_mask": np.ones((4, self.SEQ_LEN), dtype=bool),
        }

    def test_pair_save_roundtrip(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("openvino")

        model = self._model()
        compiled = compile(model, self._schema(), max_seq_len=self.SEQ_LEN)
        embeddings = export_item_embeddings(model)
        artifact = tmp_path / "artifact"
        compiled.save(artifact, item_embeddings=embeddings)

        assert (artifact / "item_embeddings.npy").is_file()
        manifest = json.loads((artifact / "manifest.json").read_text())
        assert manifest["item_matrix"]["num_items"] == self.CARDINALITY
        assert manifest["item_matrix"]["embedding_dim"] == self.EMB_DIM

        loaded = load(artifact)
        assert loaded.manifest.item_matrix["filename"] == "item_embeddings.npy"
        np.testing.assert_array_equal(load_item_embeddings(artifact).matrix, embeddings.matrix)

        inputs = self._inputs()
        expected = (
            model.forward_inference(
                feature_tensors={"item_id": torch.from_numpy(inputs["item_id"])},
                padding_mask=torch.from_numpy(inputs["padding_mask"]),
            )["logits"]
            .detach()
            .numpy()
        )
        np.testing.assert_allclose(
            loaded.predict(inputs) @ load_item_embeddings(artifact).matrix.T, expected, atol=1e-4, rtol=1e-3
        )
