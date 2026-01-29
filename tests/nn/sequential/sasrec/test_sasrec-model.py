import pytest
import torch

from replay.nn.output import InferenceOutput, TrainOutput


def test_body_forward(sasrec_model, sequential_sample):
    output = sasrec_model.body(sequential_sample["feature_tensors"], sequential_sample["padding_mask"])
    assert output.shape == (4, 7, 14)


@pytest.mark.parametrize(
    "wrong_sequential_sample",
    [
        pytest.param("missing field"),
        pytest.param("wrong length"),
        pytest.param("index out of embedding"),
    ],
    indirect=["wrong_sequential_sample"],
)
def test_wrong_input(sasrec_model, wrong_sequential_sample):
    with pytest.raises((AssertionError, IndexError, TypeError, KeyError, RuntimeError)):
        sasrec_model(**wrong_sequential_sample)


def test_sasrec_model_train_forward(tensor_schema_with_equal_embedding_dims, sasrec_model, sequential_sample):
    sasrec_model.train()
    output: TrainOutput = sasrec_model(
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


@pytest.mark.parametrize("candidates_to_score", [torch.LongTensor([1]), torch.LongTensor([0, 1, 2]), None])
def test_sasrec_inference_forward(
    tensor_schema_with_equal_embedding_dims, sasrec_model, sequential_sample, candidates_to_score
):
    sasrec_model.eval()
    output: InferenceOutput = sasrec_model(
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
