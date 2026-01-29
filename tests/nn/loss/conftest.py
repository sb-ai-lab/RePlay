import pytest

pytest.importorskip("torch")
import torch


@pytest.fixture
def hidden_simple_batch():
    seq_len = 6
    positive_labels = torch.LongTensor([[[3], [3], [2], [0], [1], [2]]])
    return {
        "model_embeddings": torch.rand(1, seq_len, 32),
        "feature_tensors": {
            "item_id": torch.LongTensor([[3, 3, 3, 2, 0, 1]]),
            "sample_weight": torch.rand_like(positive_labels, dtype=torch.float32),
        },
        "padding_mask": torch.BoolTensor([[False, False, False, True, True, True]]),
        "positive_labels": positive_labels,
        "negative_labels": torch.LongTensor([2, 0]),
        "target_padding_mask": torch.BoolTensor([[[False], [False], [True], [True], [True], [True]]]),
    }


@pytest.fixture
def hidden_simple_batch_multipositive(hidden_simple_batch):
    hidden_simple_batch["positive_labels"] = hidden_simple_batch["positive_labels"].repeat(1, 1, 5)
    hidden_simple_batch["target_padding_mask"] = hidden_simple_batch["target_padding_mask"].repeat(1, 1, 5)
    hidden_simple_batch["feature_tensors"]["sample_weight"] = \
        hidden_simple_batch["feature_tensors"]["sample_weight"].repeat(1, 1, 5)
    return hidden_simple_batch


@pytest.fixture
def hidden_simple_batch_multiclass_negatives():
    return {
        "model_embeddings": torch.rand(2, 6, 32),
        "feature_tensors": {"item_id": torch.LongTensor([[3, 3, 3, 2, 0, 1], [3, 3, 3, 3, 0, 0]])},
        "padding_mask": torch.BoolTensor(
            [[False, False, False, True, True, True], [False, False, False, False, True, True]]
        ),
        "positive_labels": torch.LongTensor([[[3], [3], [2], [0], [1], [2]], [[3], [3], [3], [0], [0], [2]]]),
        "negative_labels": torch.LongTensor([[2, 0], [1, 2]]),
        "target_padding_mask": torch.BoolTensor(
            [[[False], [False], [True], [True], [True], [True]], [[False], [False], [False], [True], [True], [True]]]
        ),
    }


@pytest.fixture
def hidden_simple_batch_multiclass_negatives_multipositive(hidden_simple_batch_multiclass_negatives):
    hidden_simple_batch_multiclass_negatives["positive_labels"] = hidden_simple_batch_multiclass_negatives[
        "positive_labels"
    ].repeat(1, 1, 5)
    hidden_simple_batch_multiclass_negatives["target_padding_mask"] = hidden_simple_batch_multiclass_negatives[
        "target_padding_mask"
    ].repeat(1, 1, 5)
    return hidden_simple_batch_multiclass_negatives
