import pytest

pytest.importorskip("torch")
import torch


@pytest.fixture
def hidden_simple_batch():
<<<<<<< HEAD
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
=======
    return {
        "model_embeddings": torch.rand(1, 6, 32),
        "feature_tensors": {"item_id": torch.LongTensor([[3, 3, 3, 2, 0, 1]])},
        "padding_mask": torch.BoolTensor([[False, False, False, True, True, True]]),
        "positive_labels": torch.LongTensor([[[3], [3], [2], [0], [1], [2]]]),
>>>>>>> Add saving/loading `linucb_arms` in pickle format
        "negative_labels": torch.LongTensor([2, 0]),
        "target_padding_mask": torch.BoolTensor([[[False], [False], [True], [True], [True], [True]]]),
    }


@pytest.fixture
def hidden_simple_batch_multipositive():
<<<<<<< HEAD
    seq_len = 6
    num_multipositives = 5
    positive_labels = torch.LongTensor([[[3], [3], [2], [0], [1], [2]]]).repeat(1, 1, num_multipositives)
    return {
        "model_embeddings": torch.rand(1, seq_len, 32),
        "feature_tensors": {
            "item_id": torch.LongTensor([[3, 3, 3, 2, 0, 1]]),
            "sample_weight": torch.rand_like(positive_labels, dtype=torch.float32),
        },
        "padding_mask": torch.BoolTensor([[False, False, False, True, True, True]]),
        "positive_labels": positive_labels,
        "negative_labels": torch.LongTensor([2, 0]),
        "target_padding_mask": torch.BoolTensor([[[False], [False], [True], [True], [True], [True]]]).repeat(
            1, 1, num_multipositives
        ),
=======
    return {
        "model_embeddings": torch.rand(1, 6, 32),
        "feature_tensors": {"item_id": torch.LongTensor([[3, 3, 3, 2, 0, 1]])},
        "padding_mask": torch.BoolTensor([[False, False, False, True, True, True]]),
        "positive_labels": torch.LongTensor([[[3], [3], [2], [0], [1], [2]]]).repeat(1, 1, 5),
        "negative_labels": torch.LongTensor([2, 0]),
        "target_padding_mask": torch.BoolTensor([[[False], [False], [True], [True], [True], [True]]]).repeat(1, 1, 5),
>>>>>>> Add saving/loading `linucb_arms` in pickle format
    }
