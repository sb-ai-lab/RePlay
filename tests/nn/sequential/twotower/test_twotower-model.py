from contextlib import nullcontext as no_exception

import pandas as pd
import pytest
import torch

from replay.nn.agg import SumAggregator
from replay.nn.embedding import SequenceEmbedding
from replay.nn.ffn import SwiGLUEncoder
from replay.nn.loss import CE
from replay.nn.mask import DefaultAttentionMask
from replay.nn.output import InferenceOutput, TrainOutput
from replay.nn.sequential import ItemReference, PositionAwareAggregator, SasRecTransformerLayer, TwoTower


def test_query_tower_forward(twotower_model, sequential_sample):
    output = twotower_model.body.query_tower(sequential_sample["feature_tensors"], sequential_sample["padding_mask"])
    assert output.shape == (
        *sequential_sample["feature_tensors"]["item_id"].shape,
        twotower_model.body.query_tower.embedding_aggregator.embedding_aggregator.embedding_dim,
    )


@pytest.mark.parametrize("candidates_to_score", [torch.LongTensor([1]), torch.LongTensor([0, 1, 2]), None])
def test_item_tower_forward(tensor_schema, twotower_model, candidates_to_score):
    output = twotower_model.body.item_tower(candidates_to_score)

    if candidates_to_score is not None:
        num_items = candidates_to_score.shape[0]
    else:
        num_items = tensor_schema["item_id"].cardinality - 1
    assert output.shape == (num_items, tensor_schema["item_id"].embedding_dim)


@pytest.mark.parametrize(
    "wrong_sequential_sample",
    [
        pytest.param("missing field"),
        pytest.param("wrong length"),
        pytest.param("index out of embedding"),
    ],
    indirect=["wrong_sequential_sample"],
)
def test_wrong_input(twotower_model, wrong_sequential_sample):
    with pytest.raises((AssertionError, IndexError, TypeError, KeyError, RuntimeError)):
        twotower_model(**wrong_sequential_sample)


@pytest.mark.parametrize("model_fixture", ["twotower_model", "twotower_model_with_context_merger"])
def test_twotower_model_train_forward(tensor_schema, request, model_fixture, sequential_sample):
    model = request.getfixturevalue(model_fixture)
    model.train()
    output: TrainOutput = model(
        feature_tensors=sequential_sample["feature_tensors"],
        padding_mask=sequential_sample["padding_mask"],
        positive_labels=sequential_sample["positive_labels"],
        target_padding_mask=sequential_sample["target_padding_mask"],
    )

    assert output["loss"].ndim == 0
    assert output["hidden_states"][0].size() == (
        *sequential_sample["feature_tensors"]["item_id"].shape,
        tensor_schema["item_id"].embedding_dim,
    )


@pytest.mark.parametrize("model_fixture", ["twotower_model", "twotower_model_with_context_merger"])
@pytest.mark.parametrize("candidates_to_score", [torch.LongTensor([1]), torch.LongTensor([0, 1, 2]), None])
def test_twotower_inference_forward(tensor_schema, request, model_fixture, sequential_sample, candidates_to_score):
    model = request.getfixturevalue(model_fixture)
    model.eval()
    output: InferenceOutput = model(
        sequential_sample["feature_tensors"], sequential_sample["padding_mask"], candidates_to_score
    )

    if candidates_to_score is not None:
        num_items = candidates_to_score.shape[0]
    else:
        num_items = tensor_schema["item_id"].cardinality - 1

    assert output["logits"].size() == (sequential_sample["padding_mask"].shape[0], num_items)
    assert output["hidden_states"][0].size() == (
        *sequential_sample["feature_tensors"]["item_id"].shape,
        tensor_schema["item_id"].embedding_dim,
    )


@pytest.mark.parametrize(
    "query_tower_names, item_tower_names, expected_exception",
    [
        (["item_id"], ["item_id"], no_exception()),
        (["item_id"], ["num_feature"], no_exception()),
        (["num_feature"], ["num_list_feature"], no_exception()),
        (None, ["item_id"], pytest.raises(TypeError)),
        (["wrong_name"], ["item_id"], pytest.raises(ValueError)),
    ],
)
def test_twotower_with_different_tower_features(
    tensor_schema, item_features_path, query_tower_names, item_tower_names, expected_exception
):
    with expected_exception:
        TwoTower(
            schema=tensor_schema,
            embedder=SequenceEmbedding(schema=tensor_schema),
            attn_mask_builder=DefaultAttentionMask(
                reference_feature_name=tensor_schema.item_id_feature_name,
                num_heads=1,
            ),
            query_tower_feature_names=query_tower_names,
            item_tower_feature_names=item_tower_names,
            query_embedding_aggregator=PositionAwareAggregator(
                embedding_aggregator=SumAggregator(64),
                max_sequence_length=7,
                dropout=0.2,
            ),
            item_embedding_aggregator=SumAggregator(64),
            query_encoder=SasRecTransformerLayer(embedding_dim=64, num_heads=1, num_blocks=1, dropout=0.2),
            query_tower_output_normalization=torch.nn.LayerNorm(64),
            item_encoder=SwiGLUEncoder(embedding_dim=64, hidden_dim=2 * 64),
            item_reference_path=item_features_path,
            loss=CE(padding_idx=tensor_schema.item_id_features.item().padding_value),
        )


def test_item_reference(tensor_schema, item_features_path):
    original_items_df = pd.read_parquet(item_features_path)

    item_reference = ItemReference(tensor_schema, item_features_path)

    assert set(item_reference.keys()) == set(original_items_df.columns)
    assert tensor_schema.item_id_feature_name in item_reference
    assert "wrong_feature" not in item_reference
    assert (
        len(item_reference[tensor_schema.item_id_feature_name])
        == original_items_df[tensor_schema.item_id_feature_name].shape[0]
    )
