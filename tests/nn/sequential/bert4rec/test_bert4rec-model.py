import pytest
import torch

from replay.nn.output import InferenceOutput, TrainOutput


def test_body_forward(bert4rec_model, bert4rec_sequential_sample):
    output = bert4rec_model.body(
        bert4rec_sequential_sample["feature_tensors"],
        bert4rec_sequential_sample["padding_mask"],
    )
    assert output.shape == (4, 5, 14)


@pytest.mark.parametrize(
    "wrong_bert4rec_sequential_sample",
    [
        pytest.param("missing field"),
        pytest.param("wrong length"),
        pytest.param("index out of embedding"),
    ],
    indirect=["wrong_bert4rec_sequential_sample"],
)
def test_wrong_input(bert4rec_model, wrong_bert4rec_sequential_sample):
    with pytest.raises((AssertionError, IndexError, TypeError, KeyError, RuntimeError)):
        bert4rec_model(**wrong_bert4rec_sequential_sample)


def test_bert4rec_model_train_forward(
    tensor_schema_with_equal_embedding_dims, bert4rec_model, bert4rec_sequential_sample
):
    bert4rec_model.train()
    output: TrainOutput = bert4rec_model(
        feature_tensors=bert4rec_sequential_sample["feature_tensors"],
        padding_mask=bert4rec_sequential_sample["padding_mask"],
        positive_labels=bert4rec_sequential_sample["positive_labels"],
        target_padding_mask=bert4rec_sequential_sample["target_padding_mask"],
    )

    assert output["loss"].ndim == 0
    assert output["hidden_states"][0].size() == (
        *bert4rec_sequential_sample["feature_tensors"]["item_id"].shape,
        tensor_schema_with_equal_embedding_dims["item_id"].embedding_dim,
    )


@pytest.mark.parametrize("candidates_to_score", [torch.LongTensor([1]), torch.LongTensor([0, 1, 2]), None])
def test_bert4rec_inference_forward(
    tensor_schema_with_equal_embedding_dims,
    bert4rec_model,
    bert4rec_sequential_sample,
    candidates_to_score,
):
    bert4rec_model.eval()
    output: InferenceOutput = bert4rec_model(
        bert4rec_sequential_sample["feature_tensors"],
        bert4rec_sequential_sample["padding_mask"],
        candidates_to_score,
    )

    if candidates_to_score is not None:
        num_items = candidates_to_score.shape[0]
    else:
        num_items = tensor_schema_with_equal_embedding_dims["item_id"].cardinality

    assert output["logits"].size() == (bert4rec_sequential_sample["padding_mask"].shape[0], num_items)
    assert output["hidden_states"][0].size() == (
        *bert4rec_sequential_sample["feature_tensors"]["item_id"].shape,
        tensor_schema_with_equal_embedding_dims["item_id"].embedding_dim,
    )
