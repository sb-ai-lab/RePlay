import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from replay.data import FeatureHint, FeatureSource, FeatureType
from replay.data.nn import (
    PandasSequentialDataset,
    TensorFeatureInfo,
    TensorFeatureSource,
    TensorSchema,
    TorchSequentialDataset,
)
from replay.nn.compiled import encode_inputs
from replay.nn.compiled._manifest import ExportManifest

pytestmark = [pytest.mark.torch]

ITEM_ID = {
    "name": "item_id",
    "dtype": "int64",
    "is_categorical": True,
    "is_list": False,
    "is_seq": True,
    "cardinality": 50,
    "tensor_dim": None,
    "embedding_dim": 32,
    "padding_value": 50,
}
PRICE = {
    "name": "price",
    "dtype": "float32",
    "is_categorical": False,
    "is_list": False,
    "is_seq": True,
    "cardinality": None,
    "tensor_dim": 1,
    "embedding_dim": 32,
    "padding_value": 0.0,
}
CAT_LIST = {
    "name": "genre",
    "dtype": "int64",
    "is_categorical": True,
    "is_list": True,
    "is_seq": True,
    "cardinality": 8,
    "tensor_dim": None,
    "embedding_dim": 32,
    "padding_value": 8,
}
ITEM_MAPPING = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8}
GENRE_MAPPING = {"x": 0, "y": 1, "z": 2}


def make_manifest(features, *, exporter="dynamo", max_seq_len=8, input_shapes=None):
    shapes = dict(input_shapes or {})
    for record in features:
        has_width = record["is_list"] or (record["tensor_dim"] or 1) > 1
        shapes.setdefault(record["name"], ["batch", "seq_len", 3] if has_width else ["batch", "seq_len"])
    shapes.setdefault("padding_mask", ["batch", "seq_len"])
    return {
        "model_type": "sasrec",
        "exporter": exporter,
        "opset_version": 20,
        "input_kinds": ["feature"] * len(features) + ["padding_mask"],
        "input_names": [record["name"] for record in features] + ["padding_mask"],
        "output_names": ["logits"],
        "candidates_policy": "full_vocab",
        "features": features,
        "input_shapes": shapes,
        "versions": {},
        "max_seq_len": max_seq_len,
        "format_version": 1,
        "item_matrix": None,
        "encoding": {"padding_side": "front", "keep": "last", "default_value": "padding_value", "mask_true": "valid"},
    }


def event(item, price=1.0, genre=None):
    return {"item_id": item, "price": price, "genre": genre if genre is not None else ["x"]}


def test_single_user_events_payload():
    manifest = make_manifest([ITEM_ID, PRICE, CAT_LIST])
    events = [event("a", 0.5), event("b", 1.5), event("unknown", genre=["y", "w"])]
    inputs = encode_inputs(manifest, {"events": events}, {"item_id": ITEM_MAPPING, "genre": GENRE_MAPPING})

    assert set(inputs) == {"item_id", "price", "genre", "padding_mask"}
    assert inputs["item_id"].dtype == np.int64
    assert inputs["price"].dtype == np.float32
    np.testing.assert_array_equal(inputs["item_id"], [[1, 2, 50]])
    np.testing.assert_allclose(inputs["price"], [[0.5, 1.5, 1.0]])
    np.testing.assert_array_equal(inputs["genre"], [[[8, 8, 0], [8, 8, 0], [8, 1, 8]]])
    np.testing.assert_array_equal(inputs["padding_mask"], [[True, True, True]])


def test_batch_front_pads_and_keeps_last_events():
    manifest = make_manifest([ITEM_ID], max_seq_len=5)
    histories = [
        [event("a"), event("b")],
        [event(key) for key in "abcdefgh"],
    ]
    inputs = encode_inputs(manifest, histories, {"item_id": ITEM_MAPPING})

    np.testing.assert_array_equal(inputs["item_id"][0], [50, 50, 50, 1, 2])
    np.testing.assert_array_equal(inputs["item_id"][1], [4, 5, 6, 7, 8])
    np.testing.assert_array_equal(inputs["padding_mask"][0], [False, False, False, True, True])
    np.testing.assert_array_equal(inputs["padding_mask"][1], [True, True, True, True, True])


def test_missing_values_fall_back_to_padding():
    manifest = make_manifest([ITEM_ID, PRICE, CAT_LIST])
    events = [{"item_id": "a"}, {"price": 2.0}, {"item_id": "b", "price": 3.0, "genre": None}]
    inputs = encode_inputs(manifest, [events], {"item_id": ITEM_MAPPING, "genre": GENRE_MAPPING})

    np.testing.assert_array_equal(inputs["item_id"], [[1, 50, 2]])
    np.testing.assert_allclose(inputs["price"], [[0.0, 2.0, 3.0]])
    np.testing.assert_array_equal(inputs["genre"][0, 2], [8, 8, 8])


def test_list_rows_normalized_to_compile_time_width():
    manifest = make_manifest([CAT_LIST])
    events = [event("a", genre=["x", "x", "x", "y"]), event("b", genre=["z"])]
    inputs = encode_inputs(manifest, [events], {"genre": GENRE_MAPPING})

    np.testing.assert_array_equal(inputs["genre"], [[[0, 0, 0], [8, 8, 2]]])


def test_numerical_list_rows_padded_to_tensor_dim():
    num_list = {
        "name": "emb",
        "dtype": "float32",
        "is_categorical": False,
        "is_list": True,
        "is_seq": True,
        "cardinality": None,
        "tensor_dim": 3,
        "embedding_dim": 32,
        "padding_value": 0.0,
    }
    manifest = make_manifest([num_list])
    events = [{"emb": [1.0, 2.0, 3.0, 4.0]}, {"emb": None}]
    inputs = encode_inputs(manifest, [events])

    np.testing.assert_allclose(inputs["emb"], [[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]])
    assert inputs["emb"].shape == (1, 2, 3)


def test_seq_len_override_and_cap():
    manifest = make_manifest([ITEM_ID], max_seq_len=4)
    histories = [[event("a"), event("b"), event("c")]]

    inputs = encode_inputs(manifest, histories, {"item_id": ITEM_MAPPING}, seq_len=4)
    np.testing.assert_array_equal(inputs["item_id"], [[50, 1, 2, 3]])
    np.testing.assert_array_equal(inputs["padding_mask"], [[False, True, True, True]])

    with pytest.raises(ValueError, match="exceeds the manifest max_seq_len"):
        encode_inputs(manifest, histories, {"item_id": ITEM_MAPPING}, seq_len=5)


def test_legacy_graph_forces_traced_length():
    manifest = make_manifest([ITEM_ID], exporter="legacy", max_seq_len=6)
    histories = [[event("a"), event("b")]]

    inputs = encode_inputs(manifest, histories, {"item_id": ITEM_MAPPING})
    assert inputs["item_id"].shape == (1, 6)
    np.testing.assert_array_equal(inputs["padding_mask"], [[False] * 4 + [True] * 2])

    inputs = encode_inputs(manifest, histories, {"item_id": ITEM_MAPPING}, seq_len=6)
    assert inputs["item_id"].shape == (1, 6)

    with pytest.raises(ValueError, match=r"[Ll]egacy"):
        encode_inputs(manifest, histories, {"item_id": ITEM_MAPPING}, seq_len=3)


def test_static_batch_size_enforced():
    manifest = make_manifest([ITEM_ID], input_shapes={"item_id": [4, "seq_len"], "padding_mask": [4, "seq_len"]})
    histories = [[event("a")], [event("b")]]

    with pytest.raises(ValueError, match="fixed batch size 4"):
        encode_inputs(manifest, histories, {"item_id": ITEM_MAPPING})

    inputs = encode_inputs(manifest, histories * 2, {"item_id": ITEM_MAPPING})
    assert inputs["item_id"].shape[0] == 4


def test_static_sequence_length_enforced():
    manifest = make_manifest([ITEM_ID], input_shapes={"item_id": ["batch", 3]})
    with pytest.raises(ValueError, match="fixed sequence length 3"):
        encode_inputs(manifest, [[event("a"), event("b")]], {"item_id": ITEM_MAPPING})
    inputs = encode_inputs(manifest, [[event("a"), event("b")]], {"item_id": ITEM_MAPPING}, seq_len=3)
    assert inputs["item_id"].shape == (1, 3)


def test_invalid_arguments():
    manifest = make_manifest([ITEM_ID])

    with pytest.raises(ValueError, match="at least one user"):
        encode_inputs(manifest, [])
    with pytest.raises(ValueError, match="Batch must be a list"):
        encode_inputs(manifest, 42)
    with pytest.raises(ValueError, match="non-empty list"):
        encode_inputs(manifest, [[]])
    with pytest.raises(ValueError, match="event must be a dict"):
        encode_inputs(manifest, [["a"]])
    with pytest.raises(ValueError, match="id mapping"):
        encode_inputs(manifest, [[event("a")]], mappings={})
    with pytest.raises(TypeError, match="ExportManifest"):
        encode_inputs(42, [[event("a")]])


def test_rejects_unsupported_contracts():
    manifest = make_manifest([ITEM_ID])
    manifest["encoding"]["padding_side"] = "back"
    with pytest.raises(ValueError, match="padding_side"):
        encode_inputs(manifest, [[event("a")]], {"item_id": ITEM_MAPPING})

    manifest = make_manifest([ITEM_ID])
    del manifest["max_seq_len"]
    with pytest.raises(ValueError, match="predates the encoding contract"):
        encode_inputs(manifest, [[event("a")]], {"item_id": ITEM_MAPPING})

    manifest = make_manifest([ITEM_ID])
    del manifest["input_shapes"]["item_id"]
    with pytest.raises(ValueError, match="no input shape"):
        encode_inputs(manifest, [[event("a")]], {"item_id": ITEM_MAPPING})


def test_non_sequential_feature_rejected():
    feature = dict(ITEM_ID, is_seq=False)
    manifest = make_manifest([feature])
    with pytest.raises(NotImplementedError, match="not sequential"):
        encode_inputs(manifest, [[event("a")]], {"item_id": ITEM_MAPPING})


def test_manifest_object_and_dict_paths_agree(tmp_path):
    data = make_manifest([ITEM_ID], max_seq_len=6)
    object_manifest = ExportManifest.from_dict(data)
    assert object_manifest.to_dict() == data
    assert object_manifest.max_seq_len == 6
    assert object_manifest.encoding.padding_side == "front"
    assert object_manifest.features[0].padding_value == 50

    histories = [[event("a"), event("b")]]
    from_dict = encode_inputs(data, histories, {"item_id": ITEM_MAPPING})
    from_object = encode_inputs(object_manifest, histories, {"item_id": ITEM_MAPPING})
    for name in from_dict:
        np.testing.assert_array_equal(from_dict[name], from_object[name])

    saved = tmp_path / "manifest.json"
    saved.write_text(json.dumps(data), encoding="utf-8")
    from_json = encode_inputs(json.loads(saved.read_text()), histories, {"item_id": ITEM_MAPPING})
    np.testing.assert_array_equal(from_json["item_id"], from_dict["item_id"])


def _parity_schema():
    return TensorSchema(
        [
            TensorFeatureInfo(
                "item_id",
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                embedding_dim=32,
                padding_value=50,
                cardinality=50,
                feature_hint=FeatureHint.ITEM_ID,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")],
            ),
            TensorFeatureInfo(
                "price",
                is_seq=True,
                feature_type=FeatureType.NUMERICAL,
                tensor_dim=1,
                padding_value=0.0,
                embedding_dim=32,
                feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "price")],
            ),
        ]
    )


def test_matches_torch_sequential_dataset_semantics():
    schema = _parity_schema()
    sequences = pd.DataFrame(
        [
            (0, [1, 2], [0.5, 1.5]),
            (1, [3, 4, 5, 6, 7, 8, 9, 10], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            (2, [7], [2.0]),
        ],
        columns=["user_id", "item_id", "price"],
    )
    sequential = PandasSequentialDataset(
        tensor_schema=schema, sequences=sequences, query_id_column="user_id", item_id_column="item_id"
    )
    with pytest.warns(DeprecationWarning):
        torch_dataset = TorchSequentialDataset(sequential, max_sequence_length=6)

    manifest = make_manifest([ITEM_ID, PRICE], max_seq_len=6)
    histories = [
        [{"item_id": item, "price": price} for item, price in zip(row.item_id, row.price)]
        for _, row in sequences.iterrows()
    ]
    identity = {value: value for value in range(50)}
    encoded = encode_inputs(manifest, histories, {"item_id": identity})

    assert len(torch_dataset) == len(histories)
    for index in range(len(torch_dataset)):
        _, mask, features = torch_dataset[index]
        np.testing.assert_array_equal(encoded["item_id"][index], features["item_id"].numpy())
        np.testing.assert_array_equal(encoded["price"][index], features["price"].numpy())
        np.testing.assert_array_equal(encoded["padding_mask"][index], mask.numpy())


ATOL, RTOL = 1e-4, 1e-3
SEQ_LEN, CARDINALITY, EMB_DIM, NUM_HEADS, NUM_BLOCKS = 16, 50, 32, 2, 1
LIST_CARDINALITY, LIST_LENGTH = 8, 3

if torch.cuda.is_available():  # D6 contract: comparisons run with TF32 off
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _e2e_schema():
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
        TensorFeatureInfo(
            "price",
            is_seq=True,
            feature_type=FeatureType.NUMERICAL,
            tensor_dim=1,
            padding_value=0.0,
            embedding_dim=EMB_DIM,
            feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "price")],
        ),
        TensorFeatureInfo(
            "genre",
            is_seq=True,
            feature_type=FeatureType.CATEGORICAL_LIST,
            embedding_dim=EMB_DIM,
            cardinality=LIST_CARDINALITY,
            padding_value=LIST_CARDINALITY,
            feature_sources=[TensorFeatureSource(FeatureSource.ITEM_FEATURES, "genre")],
        ),
    ]
    return TensorSchema(features)


def _build_sasrec(schema):
    from replay.nn.agg import SumAggregator
    from replay.nn.compiled import compile
    from replay.nn.embedding import SequenceEmbedding
    from replay.nn.loss import CE
    from replay.nn.mask import DefaultAttentionMask
    from replay.nn.sequential.sasrec.agg import PositionAwareAggregator
    from replay.nn.sequential.sasrec.model import SasRec, SasRecBody
    from replay.nn.sequential.sasrec.transformer import SasRecTransformerLayer

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
        output_normalization=torch.nn.LayerNorm(EMB_DIM),
    )
    model = SasRec(body=body, loss=CE(ignore_index=schema.item_id_features.item().padding_value))
    model.eval()
    compiled = compile(model, schema, max_seq_len=SEQ_LEN, list_feature_length=LIST_LENGTH)
    return model, compiled


def _raw_histories():
    return [
        [event("a", 0.5, ["x", "y"]), event("b", 1.0), event("c", 2.0, ["z", "w", "x", "y"])],
        [event(key, float(index), ["x"]) for index, key in enumerate("cdefghij")],
    ]


def _torch_reference(model, inputs):
    feature_tensors = {name: torch.from_numpy(array) for name, array in inputs.items() if name != "padding_mask"}
    return (
        model.forward_inference(
            feature_tensors=feature_tensors,
            padding_mask=torch.from_numpy(inputs["padding_mask"]),
        )["logits"]
        .detach()
        .numpy()
    )


@pytest.mark.conditional
def test_encoded_inputs_predict_like_torch():
    pytest.importorskip("onnx")
    pytest.importorskip("openvino")

    model, compiled = _build_sasrec(_e2e_schema())
    inputs = encode_inputs(compiled.manifest, _raw_histories(), {"item_id": ITEM_MAPPING, "genre": GENRE_MAPPING})
    assert inputs["item_id"].shape == (2, SEQ_LEN)

    np.testing.assert_allclose(compiled.predict(inputs), _torch_reference(model, inputs), atol=ATOL, rtol=RTOL)


@pytest.mark.conditional
def test_saved_artifact_dict_manifest_serving_path(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("openvino")

    from replay.nn.compiled import load

    model, compiled = _build_sasrec(_e2e_schema())
    artifact = tmp_path / "artifact"
    compiled.save(artifact)

    manifest_dict = json.loads((artifact / "manifest.json").read_text())
    assert manifest_dict["max_seq_len"] == SEQ_LEN
    inputs = encode_inputs(manifest_dict, _raw_histories(), {"item_id": ITEM_MAPPING, "genre": GENRE_MAPPING})
    expected = _torch_reference(model, inputs)

    np.testing.assert_allclose(load(artifact).predict(inputs), expected, atol=ATOL, rtol=RTOL)
