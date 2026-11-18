from typing import ClassVar

import numpy as np
import pytest
import torch

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.data import FeatureHint, FeatureSource, FeatureType
    from replay.data.nn import TensorFeatureInfo, TensorFeatureSource, TensorSchema
    from replay.nn.agg import SumAggregator
    from replay.nn.compiled import CompiledModel, capture_config, compile, load, save_config
    from replay.nn.compiled._exporters import DYNAMO, LEGACY
    from replay.nn.embedding import SequenceEmbedding
    from replay.nn.ffn import SwiGLUEncoder
    from replay.nn.loss import CE
    from replay.nn.mask import DefaultAttentionMask
    from replay.nn.sequential import TwoTower, TwoTowerBody
    from replay.nn.sequential.sasrec.agg import PositionAwareAggregator
    from replay.nn.sequential.sasrec.model import SasRec, SasRecBody
    from replay.nn.sequential.sasrec.transformer import SasRecTransformerLayer

pytestmark = pytest.mark.conditional

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("openvino")

ATOL, RTOL = 1e-4, 1e-3
BATCH, SEQ_LEN, CARDINALITY, EMB_DIM = 4, 16, 50, 32
NUM_HEADS, NUM_BLOCKS = 2, 1


def make_schema(with_list_feature=False, with_num_feature=True):
    features = [
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
    if with_num_feature:
        features.append(
            TensorFeatureInfo(
                "price",
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                tensor_dim=1,
                padding_value=0.0,
                embedding_dim=EMB_DIM,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "price")],
            ),
        )
    if with_list_feature:
        features.append(
            TensorFeatureInfo(
                "cat_list",
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL_LIST,
                embedding_dim=EMB_DIM,
                cardinality=8,
                padding_value=8,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "cat_list")],
            ),
        )
    return TensorSchema(features)


def build_sasrec(schema, normalization):
    torch.manual_seed(2026)
    body = SasRecBody(
        embedder=SequenceEmbedding(schema=schema),
        embedding_aggregator=PositionAwareAggregator(
            embedding_aggregator=SumAggregator(embedding_dim=EMB_DIM),
            max_sequence_length=SEQ_LEN,
            dropout=0.0,
        ),
        attn_mask_builder=DefaultAttentionMask(schema.item_id_feature_name, NUM_HEADS),
        encoder=SasRecTransformerLayer(
            embedding_dim=EMB_DIM,
            num_heads=NUM_HEADS,
            num_blocks=NUM_BLOCKS,
            dropout=0.0,
        ),
        output_normalization=normalization,
    )
    return SasRec(body=body, loss=CE(ignore_index=schema.item_id_features.item().padding_value))


class _StubFeaturesReader:
    feature_names: ClassVar[list[str]] = []


def build_twotower(schema):
    torch.manual_seed(2026)
    body = TwoTowerBody(
        schema=schema,
        embedder=SequenceEmbedding(schema=schema),
        attn_mask_builder=DefaultAttentionMask(schema.item_id_feature_name, NUM_HEADS),
        query_tower_feature_names=schema.names,
        query_embedding_aggregator=PositionAwareAggregator(
            embedding_aggregator=SumAggregator(embedding_dim=EMB_DIM),
            max_sequence_length=SEQ_LEN,
            dropout=0.0,
        ),
        item_embedding_aggregator=SumAggregator(embedding_dim=EMB_DIM),
        query_encoder=SasRecTransformerLayer(
            embedding_dim=EMB_DIM,
            num_heads=NUM_HEADS,
            num_blocks=NUM_BLOCKS,
            dropout=0.0,
        ),
        query_tower_output_normalization=torch.nn.LayerNorm(EMB_DIM),
        item_encoder=SwiGLUEncoder(embedding_dim=EMB_DIM, hidden_dim=2 * EMB_DIM),
        item_features_reader=_StubFeaturesReader(),
    )
    return TwoTower(body=body, loss=CE(ignore_index=schema.item_id_features.item().padding_value))


def make_inputs(batch=BATCH, seq_len=SEQ_LEN, num_candidates=None):
    rng = np.random.RandomState(2026)
    inputs = {
        "item_id": rng.randint(0, CARDINALITY, size=(batch, seq_len)).astype(np.int64),
        "price": rng.standard_normal((batch, seq_len)).astype(np.float32),
        "padding_mask": np.ones((batch, seq_len), dtype=bool),
    }
    inputs["padding_mask"][0, : seq_len // 3] = False
    inputs["padding_mask"][2, : 2 * seq_len // 3] = False
    if num_candidates is not None:
        inputs["candidates"] = np.arange(num_candidates, dtype=np.int64)
    return inputs


def twotower_query_reference(model, inputs):
    model.eval()
    feature_tensors = {
        name: torch.from_numpy(array) for name, array in inputs.items() if name not in ("padding_mask", "candidates")
    }
    with torch.no_grad():
        hidden_states = model.body.query_tower(feature_tensors, torch.from_numpy(inputs["padding_mask"]))
        if model.context_merger is not None:
            hidden_states = model.context_merger(
                model_hidden_state=hidden_states,
                feature_tensors=feature_tensors,
            )
        return hidden_states[:, -1, :].contiguous().numpy()


def torch_reference(model, inputs):
    model.eval()
    feature_tensors = {
        name: torch.from_numpy(array) for name, array in inputs.items() if name not in ("padding_mask", "candidates")
    }
    candidates = inputs.get("candidates")
    with torch.no_grad():
        output = model.forward_inference(
            feature_tensors=feature_tensors,
            padding_mask=torch.from_numpy(inputs["padding_mask"]),
            candidates_to_score=torch.from_numpy(candidates) if candidates is not None else None,
        )
    return output["logits"].numpy()


def assert_parity(compiled, model, inputs):
    expected = torch_reference(model, inputs)
    predicted = compiled.predict(inputs)
    assert predicted.shape == expected.shape
    np.testing.assert_allclose(predicted, expected, atol=ATOL, rtol=RTOL)


@pytest.fixture(scope="module")
def tensor_schema():
    return make_schema()


@pytest.fixture(scope="module")
def sasrec(tensor_schema):
    return build_sasrec(tensor_schema, torch.nn.LayerNorm(EMB_DIM))


@pytest.fixture(scope="module")
def sasrec_rmsnorm(tensor_schema):
    return build_sasrec(tensor_schema, torch.nn.RMSNorm(EMB_DIM))


@pytest.fixture(scope="module")
def twotower(tensor_schema):
    return build_twotower(tensor_schema)


@pytest.fixture(scope="module")
def inputs():
    return make_inputs()


def test_full_vocab_parity_and_batch_growth(sasrec, tensor_schema, inputs):
    compiled = compile(sasrec, tensor_schema, max_seq_len=SEQ_LEN)
    assert isinstance(compiled, CompiledModel)
    assert compiled.manifest.exporter == LEGACY
    assert compiled.manifest.candidates_policy == "full_vocab"
    assert compiled.input_names == ["item_id", "price", "padding_mask"]
    assert_parity(compiled, sasrec, inputs)

    smaller = {name: array[:2] for name, array in inputs.items()}
    assert_parity(compiled, sasrec, smaller)


def test_sequence_length_is_dynamic(sasrec, tensor_schema, inputs):
    # dynamo honors runtime sequence length; legacy traces shape math as constants
    compiled = compile(sasrec, tensor_schema, exporter="dynamo", max_seq_len=SEQ_LEN)
    shorter = {
        "item_id": inputs["item_id"][:, -10:],
        "price": inputs["price"][:, -10:],
        "padding_mask": inputs["padding_mask"][:, -10:],
    }
    assert_parity(compiled, sasrec, shorter)


def test_mode_batch_keeps_static_batch(sasrec, tensor_schema, inputs):
    compiled = compile(sasrec, tensor_schema, mode="batch", batch_size=BATCH, max_seq_len=SEQ_LEN)
    assert compiled.manifest.input_shapes["item_id"] == [BATCH, "seq_len"]
    assert compiled.manifest.input_shapes["padding_mask"] == [BATCH, "seq_len"]
    assert_parity(compiled, sasrec, inputs)


def test_explicit_candidates(sasrec, tensor_schema):
    inputs = make_inputs(num_candidates=6)
    compiled = compile(sasrec, tensor_schema, candidates_policy="explicit", num_candidates=6, max_seq_len=SEQ_LEN)
    assert compiled.manifest.candidates_policy == "explicit"
    assert "candidates" in compiled.input_names
    assert_parity(compiled, sasrec, inputs)

    permuted = dict(inputs)
    permuted["candidates"] = np.array([3, 1, 4, 5, 0, 2], dtype=np.int64)
    assert_parity(compiled, sasrec, permuted)


def test_rmsnorm_auto_selects_dynamo(sasrec_rmsnorm, tensor_schema, inputs):
    compiled = compile(sasrec_rmsnorm, tensor_schema, max_seq_len=SEQ_LEN)
    assert compiled.manifest.exporter == DYNAMO
    assert_parity(compiled, sasrec_rmsnorm, inputs)


def test_explicit_exporter_override(sasrec, tensor_schema, inputs):
    compiled = compile(sasrec, tensor_schema, exporter="dynamo", max_seq_len=SEQ_LEN)
    assert compiled.manifest.exporter == DYNAMO
    assert_parity(compiled, sasrec, inputs)


def test_num_threads(sasrec, tensor_schema, inputs):
    compiled = compile(sasrec, tensor_schema, num_threads=2, max_seq_len=SEQ_LEN)
    assert_parity(compiled, sasrec, inputs)


def test_twotower_query_graph(twotower, tensor_schema, inputs):
    compiled = compile(twotower, tensor_schema, max_seq_len=SEQ_LEN)
    assert compiled.manifest.model_type == "twotower_query"
    assert compiled.manifest.output_names == ["query_embeddings"]
    predicted = compiled.predict(inputs)
    expected = twotower_query_reference(twotower, inputs)
    assert predicted.shape == expected.shape
    np.testing.assert_allclose(predicted, expected, atol=ATOL, rtol=RTOL)


def test_save_load_roundtrip(sasrec, tensor_schema, inputs, tmp_path):
    compiled = compile(sasrec, tensor_schema, max_seq_len=SEQ_LEN)
    artifact = tmp_path / "artifact"
    compiled.save(artifact)
    assert (artifact / "model.onnx").is_file()
    assert (artifact / "manifest.json").is_file()

    loaded = load(artifact)
    assert loaded.manifest.model_type == compiled.manifest.model_type
    assert loaded.manifest.exporter == compiled.manifest.exporter
    assert loaded.manifest.features == compiled.manifest.features
    assert loaded.manifest.input_shapes == compiled.manifest.input_shapes
    assert set(loaded.manifest.versions) == {"torch", "onnx", "openvino"}
    assert_parity(loaded, sasrec, inputs)
    assert repr(loaded).startswith("CompiledModel(model_type=")


def test_load_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


def test_unknown_mode(sasrec, tensor_schema):
    with pytest.raises(ValueError, match="Unknown mode"):
        compile(sasrec, tensor_schema, mode="one_query")


def test_unknown_exporter(sasrec, tensor_schema):
    with pytest.raises(ValueError, match="Unknown exporter"):
        compile(sasrec, tensor_schema, exporter="onnx")


def test_explicit_policy_requires_num_candidates(sasrec, tensor_schema):
    with pytest.raises(ValueError, match="num_candidates is required"):
        compile(sasrec, tensor_schema, candidates_policy="explicit")


def test_full_vocab_rejects_num_candidates(sasrec, tensor_schema):
    with pytest.raises(ValueError, match="num_candidates is only accepted"):
        compile(sasrec, tensor_schema, num_candidates=5)


def test_unknown_candidates_policy(sasrec, tensor_schema):
    with pytest.raises(ValueError, match="Unknown candidates policy"):
        compile(sasrec, tensor_schema, candidates_policy="half_vocab")


def test_twotower_rejects_explicit_candidates(twotower, tensor_schema):
    with pytest.raises(ValueError, match="query-tower graph scores the external item matrix"):
        compile(twotower, tensor_schema, candidates_policy="explicit", num_candidates=4)


def test_list_feature_not_exportable(sasrec):
    schema = make_schema(with_list_feature=True)
    with pytest.raises(NotImplementedError, match="list features are not exportable"):
        compile(sasrec, schema)


def test_unsupported_model_type():
    with pytest.raises(TypeError, match="No compiled adapter"):
        compile(torch.nn.Identity(), make_schema())


def test_predict_input_mismatch(sasrec, tensor_schema, inputs):
    compiled = compile(sasrec, tensor_schema, max_seq_len=SEQ_LEN)
    with pytest.raises(ValueError, match="missing=\\['item_id', 'price', 'padding_mask'\\]"):
        compiled.predict({})
    with pytest.raises(ValueError, match="unexpected=\\['user_id'\\]"):
        compiled.predict({**inputs, "user_id": np.zeros(1, dtype=np.int64)})


def test_predict_accepts_torch_tensors(sasrec, tensor_schema, inputs):
    compiled = compile(sasrec, tensor_schema, max_seq_len=SEQ_LEN)
    tensor_inputs = {name: torch.from_numpy(array) for name, array in inputs.items()}
    predicted = compiled.predict(tensor_inputs)
    expected = torch_reference(sasrec, inputs)
    np.testing.assert_allclose(predicted, expected, atol=ATOL, rtol=RTOL)


class _DoublingMerger(torch.nn.Module):
    def forward(self, model_hidden_state, feature_tensors):
        return model_hidden_state * 2


def test_twotower_context_merger(twotower, tensor_schema, inputs):
    twotower.context_merger = _DoublingMerger()
    compiled = compile(twotower, tensor_schema, max_seq_len=SEQ_LEN)
    predicted = compiled.predict(inputs)
    expected = twotower_query_reference(twotower, inputs)
    assert predicted.shape == expected.shape
    np.testing.assert_allclose(predicted, expected, atol=ATOL, rtol=RTOL)


def test_multi_dim_numerical_feature():
    schema = make_schema(with_num_feature=False)
    schema = TensorSchema(
        [
            *schema.values(),
            TensorFeatureInfo(
                "dims",
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                tensor_dim=3,
                padding_value=0.0,
                embedding_dim=EMB_DIM,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "dims")],
            ),
        ]
    )
    model = build_sasrec(schema, torch.nn.LayerNorm(EMB_DIM))
    rng = np.random.RandomState(7)
    inputs = {
        "item_id": rng.randint(0, CARDINALITY, size=(BATCH, SEQ_LEN)).astype(np.int64),
        "dims": rng.standard_normal((BATCH, SEQ_LEN, 3)).astype(np.float32),
        "padding_mask": np.ones((BATCH, SEQ_LEN), dtype=bool),
    }
    compiled = compile(model, schema, max_seq_len=SEQ_LEN)
    assert compiled.manifest.input_shapes["dims"] == ["batch", "seq_len", 3]
    assert_parity(compiled, model, inputs)


def test_export_seam_default_and_unknown_exporter():
    from replay.nn.compiled._exporters import export_onnx

    module = torch.nn.Linear(4, 2)
    args = (torch.randn(3, 4),)
    onnx_bytes = export_onnx(module, args, ["x"], ["y"], {}, 20)
    assert onnx_bytes
    with pytest.raises(ValueError, match="Unknown exporter"):
        export_onnx(module, args, ["x"], ["y"], {}, 20, exporter="tensorrt")


FROM_PARAMS = {
    "embedding_dim": EMB_DIM,
    "num_heads": NUM_HEADS,
    "num_blocks": NUM_BLOCKS,
    "max_sequence_length": SEQ_LEN,
    "dropout": 0.0,
}


def _save_sasrec_checkpoint(tensor_schema, tmp_path):
    model, config = capture_config(SasRec, tensor_schema, **FROM_PARAMS)
    ckpt = tmp_path / "best.ckpt"
    torch.save(model.state_dict(), ckpt)
    sidecar = save_config(config, ckpt)
    return model, ckpt, sidecar


def test_compile_from_checkpoint(tensor_schema, inputs, tmp_path):
    model, ckpt, sidecar = _save_sasrec_checkpoint(tensor_schema, tmp_path)

    compiled = compile(ckpt, max_seq_len=SEQ_LEN)
    assert compiled.manifest.model_type == "sasrec"
    assert_parity(compiled, model, inputs)

    compiled_explicit = compile(ckpt, max_seq_len=SEQ_LEN, config_path=sidecar)
    assert_parity(compiled_explicit, model, inputs)


def test_compile_checkpoint_rejects_schema(tensor_schema, tmp_path):
    _, ckpt, _ = _save_sasrec_checkpoint(tensor_schema, tmp_path)
    with pytest.raises(ValueError, match="do not pass schema"):
        compile(ckpt, tensor_schema, max_seq_len=SEQ_LEN)


def test_compile_checkpoint_missing_sidecar(tmp_path):
    ckpt = tmp_path / "orphan.ckpt"
    torch.save({}, ckpt)
    with pytest.raises(FileNotFoundError, match="config sidecar"):
        compile(ckpt, max_seq_len=SEQ_LEN)


def test_compile_module_requires_schema(sasrec):
    with pytest.raises(ValueError, match="schema is required"):
        compile(sasrec)
