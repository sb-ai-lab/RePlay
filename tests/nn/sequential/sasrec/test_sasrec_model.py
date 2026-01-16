import pytest
import torch

from replay.nn.output import InferenceOutput, TrainOutput


def test_body_forward(sasrec_model, sequential_sample):
    output = sasrec_model.body(sequential_sample["feature_tensors"], sequential_sample["padding_mask"])
    assert output.shape == (4, 7, 64)


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


def sasrec_test_model_train_forward(sasrec_model, sequential_sample):
    sasrec_model.train()
    output: TrainOutput = sasrec_model(
        sequential_sample["feature_tensors"],
        sequential_sample["padding_mask"],
        sequential_sample["positive_labels"],
        sequential_sample["target_padding_mask"],
    )

    assert output["loss"].ndim == 0
    assert output["hidden_states"][0].size() == (2, 7, 64)


@pytest.mark.parametrize(
    "candidates_to_score, expected_shape",
    [
        (torch.LongTensor([1]), (2, 1)),
        (torch.LongTensor([0, 1, 2]), (2, 3)),
        (None, (2, 3)),
    ],
)
def sasrec_test_model_inference_forward(sasrec_model, sequential_sample, candidates_to_score, expected_shape):
    sasrec_model.eval()
    output: InferenceOutput = sasrec_model(
        sequential_sample["feature_tensors"], sequential_sample["padding_mask"], candidates_to_score
    )
    assert output["logits"].size() == expected_shape
    assert output["hidden_states"][0].size() == (2, 7, 64)
