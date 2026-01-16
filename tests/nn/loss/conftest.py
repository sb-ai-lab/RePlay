import pytest

pytest.importorskip("torch")
import torch


@pytest.fixture
def hidden_simple_batch():
    return {
        "model_embeddings": torch.rand(1, 6, 32),
        "feature_tensors": {"item_id": torch.LongTensor([[3, 3, 3, 2, 0, 1]])},
        "padding_mask": torch.BoolTensor([[False, False, False, True, True, True]]),
        "positive_labels": torch.LongTensor([[[3], [3], [2], [0], [1], [2]]]),
        "negative_labels": torch.LongTensor([2, 0]),
        "target_padding_mask": torch.BoolTensor([[[False], [False], [True], [True], [True], [True]]]),
    }


@pytest.fixture
def hidden_simple_batch_multipositive():
    return {
        "model_embeddings": torch.rand(1, 6, 32),
        "feature_tensors": {"item_id": torch.LongTensor([[3, 3, 3, 2, 0, 1]])},
        "padding_mask": torch.BoolTensor([[False, False, False, True, True, True]]),
        "positive_labels": torch.LongTensor([[[3], [3], [2], [0], [1], [2]]]).repeat(1, 1, 5),
        "negative_labels": torch.LongTensor([2, 0]),
        "target_padding_mask": torch.BoolTensor([[[False], [False], [True], [True], [True], [True]]]).repeat(1, 1, 5),
    }
