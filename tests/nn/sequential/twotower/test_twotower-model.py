from contextlib import nullcontext as no_exception

import pytest
import torch

from replay.nn.agg import SumAggregator
from replay.nn.embedding import SequenceEmbedding
from replay.nn.ffn import SwiGLUEncoder
from replay.nn.mask import DefaultAttentionMask
from replay.nn.output import InferenceOutput, TrainOutput
from replay.nn.sequential import PositionAwareAggregator, SasRecTransformerLayer
from replay.nn.sequential.twotower import ItemTower, TwoTowerBody


def test_query_tower_forward(twotower_model, sequential_sample):
    output = twotower_model.body.query_tower(sequential_sample["feature_tensors"], sequential_sample["padding_mask"])
    assert output.shape == (
        *sequential_sample["feature_tensors"]["item_id"].shape,
        twotower_model.body.query_tower.embedding_aggregator.embedding_aggregator.embedding_dim,
    )


@pytest.mark.parametrize("candidates_to_score", [torch.LongTensor([1]), torch.LongTensor([0, 1, 2]), None])
def test_item_tower_forward(tensor_schema_with_equal_embedding_dims, twotower_model, candidates_to_score):
    output = twotower_model.body.item_tower(candidates_to_score)

    if candidates_to_score is not None:
        num_items = candidates_to_score.shape[0]
    else:
        num_items = tensor_schema_with_equal_embedding_dims["item_id"].cardinality
    assert output.shape == (num_items, tensor_schema_with_equal_embedding_dims["item_id"].embedding_dim)


def test_item_tower_from_checkpoint(create_twotower_model, tmp_path):
    source_model = create_twotower_model()
    source_item_tower = source_model.body.item_tower
    source_item_tower.eval()
    source_item_tower()

    checkpoint_path = tmp_path / "item_tower.ckpt"
    torch.save(source_item_tower.state_dict(), checkpoint_path)

    target_model = create_twotower_model()
    loaded_item_tower = ItemTower.from_checkpoint(
        state_dict=torch.load(checkpoint_path),
        embedder=target_model.body.embedder,
        embedding_aggregator=target_model.body.item_tower.embedding_aggregator,
        encoder=target_model.body.item_tower.encoder,
    )
    loaded_item_tower.eval()

    expected_state_dict = source_item_tower.state_dict()
    actual_state_dict = loaded_item_tower.state_dict()

    assert "cache" in actual_state_dict.keys()
    assert expected_state_dict.keys() == actual_state_dict.keys()
    for key in expected_state_dict:
        torch.testing.assert_close(actual_state_dict[key], expected_state_dict[key])


def test_item_tower_from_checkpoint_without_item_references_raises_error(create_twotower_model, tmp_path):
    source_model = create_twotower_model()
    checkpoint_path = tmp_path / "item_tower_without_item_references.ckpt"
    torch.save(source_model.body.item_tower.state_dict(), checkpoint_path)

    state_dict = torch.load(checkpoint_path)
    state_dict_without_item_references = {
        key: value for key, value in state_dict.items() if not key.startswith("item_reference_")
    }

    target_model = create_twotower_model()
    with pytest.raises(ValueError):
        ItemTower.from_checkpoint(
            state_dict=state_dict_without_item_references,
            embedder=target_model.body.embedder,
            embedding_aggregator=target_model.body.item_tower.embedding_aggregator,
            encoder=target_model.body.item_tower.encoder,
        )


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
def test_twotower_model_train_forward(
    tensor_schema_with_equal_embedding_dims, request, model_fixture, sequential_sample
):
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
        tensor_schema_with_equal_embedding_dims["item_id"].embedding_dim,
    )


@pytest.mark.parametrize("model_fixture", ["twotower_model", "twotower_model_with_context_merger"])
@pytest.mark.parametrize("candidates_to_score", [torch.LongTensor([1]), torch.LongTensor([0, 1, 2]), None])
def test_twotower_inference_forward(
    tensor_schema_with_equal_embedding_dims, request, model_fixture, sequential_sample, candidates_to_score
):
    model = request.getfixturevalue(model_fixture)
    model.eval()
    output: InferenceOutput = model(
        sequential_sample["feature_tensors"], sequential_sample["padding_mask"], candidates_to_score
    )

    if candidates_to_score is not None:
        num_items = candidates_to_score.shape[0]
    else:
        num_items = tensor_schema_with_equal_embedding_dims["item_id"].cardinality

    assert output["logits"].size() == (sequential_sample["padding_mask"].shape[0], num_items)
    assert output["hidden_states"][0].size() == (
        *sequential_sample["feature_tensors"]["item_id"].shape,
        tensor_schema_with_equal_embedding_dims["item_id"].embedding_dim,
    )


def test_twotower_inference_forward_with_trimmed_batch(
    twotower_model_only_items, sequential_sample_trimmed, tensor_schema_with_equal_embedding_dims
):
    twotower_model_only_items.eval()
    output = twotower_model_only_items(
        sequential_sample_trimmed["feature_tensors"], sequential_sample_trimmed["padding_mask"]
    )

    num_items = tensor_schema_with_equal_embedding_dims["item_id"].cardinality
    assert output["logits"].size() == (sequential_sample_trimmed["padding_mask"].shape[0], num_items)


@pytest.mark.parametrize(
    "item_features_reader, query_tower_names, expected_exception",
    [
        (["item_id"], ["item_id"], no_exception()),
        (["item_id"], ["item_id", "cat_list_feature"], no_exception()),
        (["item_id", "num_list_feature"], ["item_id", "cat_list_feature"], no_exception()),
        (["item_id"], None, pytest.raises(TypeError)),
        (["item_id"], ["wrong_name"], pytest.raises(ValueError)),
    ],
    indirect=["item_features_reader"],
)
def test_twotower_with_different_tower_features(
    tensor_schema, item_features_reader, query_tower_names, expected_exception
):
    with expected_exception:
        TwoTowerBody(
            schema=tensor_schema,
            embedder=SequenceEmbedding(schema=tensor_schema),
            attn_mask_builder=DefaultAttentionMask(
                reference_feature_name=tensor_schema.item_id_feature_name,
                num_heads=1,
            ),
            query_tower_feature_names=query_tower_names,
            query_embedding_aggregator=PositionAwareAggregator(
                embedding_aggregator=SumAggregator(64),
                max_sequence_length=7,
                dropout=0.2,
            ),
            item_embedding_aggregator=SumAggregator(64),
            query_encoder=SasRecTransformerLayer(embedding_dim=64, num_heads=1, num_blocks=1, dropout=0.2),
            query_tower_output_normalization=torch.nn.LayerNorm(64),
            item_encoder=SwiGLUEncoder(embedding_dim=64, hidden_dim=2 * 64),
            item_features_reader=item_features_reader,
        )
