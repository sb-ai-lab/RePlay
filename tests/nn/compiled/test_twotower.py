import json

import numpy as np
import pytest
import torch

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.data import FeatureHint, FeatureSource, FeatureType
    from replay.data.nn import TensorFeatureInfo, TensorFeatureSource, TensorSchema
    from replay.nn.compiled import (
        capture_config,
        compile,
        export_item_embeddings,
        load_item_embeddings,
        save_config,
    )
    from replay.nn.sequential.sasrec.model import SasRec
    from replay.nn.sequential.twotower.model import TwoTower

pytestmark = pytest.mark.conditional

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("openvino")

ATOL, RTOL = 1e-4, 1e-3
SEQ_LEN, CARDINALITY, EMB_DIM = 16, 50, 32
BATCH = 4


class _DictFeaturesReader:
    def __init__(self, features):
        self._features = features
        self.feature_names = list(features)

    def __getitem__(self, name):
        return self._features[name]


def make_reader():
    return _DictFeaturesReader({"item_id": torch.arange(CARDINALITY)})


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
        ]
    )


def build_model():
    torch.manual_seed(2026)
    model = TwoTower.from_params(
        schema=make_schema(),
        item_features_reader=make_reader(),
        embedding_dim=EMB_DIM,
        num_heads=2,
        num_blocks=1,
        max_sequence_length=SEQ_LEN,
        dropout=0.0,
    )
    model.eval()
    return model


def make_inputs(batch=BATCH, seq_len=SEQ_LEN):
    rng = np.random.RandomState(2026)
    return {
        "item_id": rng.randint(0, CARDINALITY, size=(batch, seq_len)).astype(np.int64),
        "padding_mask": np.ones((batch, seq_len), dtype=bool),
    }


def torch_logits(model, inputs, candidates=None):
    feature_tensors = {name: torch.from_numpy(array) for name, array in inputs.items() if name != "padding_mask"}
    candidates_to_score = None if candidates is None else torch.from_numpy(candidates)
    return (
        model.forward_inference(
            feature_tensors=feature_tensors,
            padding_mask=torch.from_numpy(inputs["padding_mask"]),
            candidates_to_score=candidates_to_score,
        )["logits"]
        .detach()
        .numpy()
    )


def test_export_item_embeddings_matches_item_tower():
    model = build_model()
    embeddings = export_item_embeddings(model)

    assert embeddings.num_items == CARDINALITY
    assert embeddings.embedding_dim == EMB_DIM
    assert embeddings.matrix.dtype == np.float32
    assert embeddings.feature_names == ["item_id"]

    with torch.no_grad():
        expected = model.body.item_tower._encode_item_rows(None).detach().numpy()
    np.testing.assert_allclose(embeddings.matrix, expected, atol=0, rtol=0)


def test_export_item_embeddings_rejects_other_models():
    with pytest.raises(TypeError, match="TwoTower"):
        export_item_embeddings(torch.nn.Identity())


def test_two_artifact_parity_full_vocab():
    model = build_model()
    compiled = compile(model, make_schema(), max_seq_len=SEQ_LEN)
    inputs = make_inputs()
    embeddings = export_item_embeddings(model)

    query = compiled.predict(inputs)
    assert query.shape == (BATCH, EMB_DIM)
    np.testing.assert_allclose(query @ embeddings.matrix.T, torch_logits(model, inputs), atol=ATOL, rtol=RTOL)


def test_two_artifact_parity_explicit_candidates():
    model = build_model()
    compiled = compile(model, make_schema(), max_seq_len=SEQ_LEN)
    inputs = make_inputs()
    embeddings = export_item_embeddings(model)
    candidates = np.array([3, 7, 19, 41], dtype=np.int64)

    query = compiled.predict(inputs)
    np.testing.assert_allclose(
        query @ embeddings.matrix[candidates].T,
        torch_logits(model, inputs, candidates=candidates),
        atol=ATOL,
        rtol=RTOL,
    )


def test_item_embeddings_save_load_roundtrip(tmp_path):
    model = build_model()
    embeddings = export_item_embeddings(model)
    artifact = tmp_path / "items"
    embeddings.save(artifact)

    layout = json.loads((artifact / "item_embeddings.json").read_text())
    assert layout["num_items"] == CARDINALITY
    assert layout["embedding_dim"] == EMB_DIM

    loaded = load_item_embeddings(artifact)
    np.testing.assert_array_equal(loaded.matrix, embeddings.matrix)
    assert loaded.feature_names == embeddings.feature_names

    corrupted = json.loads((artifact / "item_embeddings.json").read_text())
    corrupted["num_items"] = CARDINALITY + 1
    (artifact / "item_embeddings.json").write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="disagrees with the layout manifest"):
        load_item_embeddings(artifact)

    with pytest.raises(FileNotFoundError, match="No item embeddings artifact"):
        load_item_embeddings(tmp_path / "missing")


def test_two_artifacts_from_one_checkpoint(tmp_path):
    model, config = capture_config(
        TwoTower,
        make_schema(),
        item_features_reader=make_reader(),
        embedding_dim=EMB_DIM,
        num_heads=2,
        num_blocks=1,
        max_sequence_length=SEQ_LEN,
        dropout=0.0,
    )
    model.eval()
    ckpt = tmp_path / "twotower.ckpt"
    torch.save(model.state_dict(), ckpt)
    save_config(config, ckpt)

    inputs = make_inputs()
    expected = torch_logits(model, inputs)
    expected_candidates = torch_logits(model, inputs, candidates=np.array([1, 30], dtype=np.int64))

    from replay.nn.compiled._checkpoint import restore_from_checkpoint

    with pytest.raises(ValueError, match="item_features_reader"):
        restore_from_checkpoint(ckpt)
    restored, restored_schema = restore_from_checkpoint(ckpt, item_features_reader=make_reader())
    assert restored_schema.names == make_schema().names

    compiled = compile(restored, restored_schema, max_seq_len=SEQ_LEN)
    embeddings = export_item_embeddings(restored)
    query = compiled.predict(inputs)
    np.testing.assert_allclose(query @ embeddings.matrix.T, expected, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(query @ embeddings.matrix[[1, 30]].T, expected_candidates, atol=ATOL, rtol=RTOL)

    compiled_from_ckpt = compile(ckpt, item_features_reader=make_reader(), max_seq_len=SEQ_LEN)
    np.testing.assert_allclose(compiled_from_ckpt.predict(inputs), query, atol=ATOL, rtol=RTOL)


def test_capture_config_rejects_reader_for_sasrec():

    schema = make_schema()
    with pytest.raises(ValueError, match="only accepted for TwoTower"):
        capture_config(
            SasRec,
            schema,
            item_features_reader=make_reader(),
            embedding_dim=EMB_DIM,
            num_heads=2,
            num_blocks=1,
            max_sequence_length=SEQ_LEN,
        )


def test_compile_rejects_reader_with_module():
    model = build_model()
    with pytest.raises(ValueError, match="only accepted with a checkpoint path"):
        compile(model, make_schema(), item_features_reader=make_reader())
