import time

import numpy as np
import pytest
import torch

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.data import FeatureHint, FeatureSource, FeatureType
    from replay.data.nn import TensorFeatureInfo, TensorFeatureSource, TensorSchema
    from replay.nn.compiled import compile
    from replay.nn.sequential.sasrec.model import SasRec
    from replay.nn.sequential.twotower.model import TwoTower

pytestmark = pytest.mark.conditional

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("openvino")

BATCH = 32
WARMUP, ITERS = 5, 30
SPEEDUP_FLOOR = 1.2
SEQ_LEN, CARDINALITY, EMB_DIM = 20, 1000, 64
NUM_HEADS, NUM_BLOCKS = 2, 2

if torch.cuda.is_available():  # comparisons run with TF32 off
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


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


class _DictFeaturesReader:
    def __init__(self, features):
        self._features = features
        self.feature_names = list(features)

    def __getitem__(self, name):
        return self._features[name]


def make_inputs():
    rng = np.random.RandomState(2026)
    return {
        "item_id": rng.randint(0, CARDINALITY, size=(BATCH, SEQ_LEN)).astype(np.int64),
        "padding_mask": np.ones((BATCH, SEQ_LEN), dtype=bool),
    }


def median_ms(fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(ITERS):
        start = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - start)
    return float(np.median(times)) / 1e6


def torch_feature_inputs(inputs):
    feature_tensors = {name: torch.from_numpy(array) for name, array in inputs.items() if name != "padding_mask"}
    return feature_tensors, torch.from_numpy(inputs["padding_mask"])


def test_sasrec_compiled_latency() -> None:
    torch.manual_seed(2026)
    model = SasRec.from_params(
        schema=make_schema(),
        embedding_dim=EMB_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_sequence_length=SEQ_LEN,
        dropout=0.0,
    )
    model.eval()
    compiled = compile(model, make_schema(), max_seq_len=SEQ_LEN, batch_size=BATCH, mode="batch")
    inputs = make_inputs()
    feature_tensors, padding_mask = torch_feature_inputs(inputs)

    def torch_call():
        with torch.inference_mode():
            model.forward_inference(feature_tensors=feature_tensors, padding_mask=padding_mask)

    def compiled_call():
        compiled.predict(inputs)

    torch_ms = median_ms(torch_call)
    compiled_ms = median_ms(compiled_call)
    speedup = torch_ms / compiled_ms
    msg = (
        f"sasrec full-vocab b{BATCH} speedup {speedup:.2f}x below {SPEEDUP_FLOOR}x"
        f" (torch {torch_ms:.3f} ms, compiled {compiled_ms:.3f} ms)"
    )
    assert speedup >= SPEEDUP_FLOOR, msg


def test_twotower_query_compiled_latency() -> None:
    torch.manual_seed(2026)
    model = TwoTower.from_params(
        schema=make_schema(),
        item_features_reader=_DictFeaturesReader({"item_id": torch.arange(CARDINALITY)}),
        embedding_dim=EMB_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_sequence_length=SEQ_LEN,
        dropout=0.0,
    )
    model.eval()
    compiled = compile(model, make_schema(), max_seq_len=SEQ_LEN, batch_size=BATCH, mode="batch")
    inputs = make_inputs()
    feature_tensors, padding_mask = torch_feature_inputs(inputs)

    def torch_call():
        with torch.inference_mode():
            model.body.query_tower(feature_tensors=feature_tensors, padding_mask=padding_mask)[:, -1, :]

    def compiled_call():
        compiled.predict(inputs)

    torch_ms = median_ms(torch_call)
    compiled_ms = median_ms(compiled_call)
    speedup = torch_ms / compiled_ms
    msg = (
        f"twotower query b{BATCH} speedup {speedup:.2f}x below {SPEEDUP_FLOOR}x"
        f" (torch {torch_ms:.3f} ms, compiled {compiled_ms:.3f} ms)"
    )
    assert speedup >= SPEEDUP_FLOOR, msg
