import pytest
import torch
from replay.nn.head import EmbeddingTyingHead


@pytest.mark.parametrize(
    "shape_hidden, shape_embeddings, expected_shape",
    [
        ((25, 1, 128), (1000, 128), (25, 1, 1000)),
        ((25, 50, 128), (1000, 128), (25, 50, 1000)),
        ((25, 128), (25, 1000, 128), (25, 1000)),
        ((25, 50, 128), (25, 50, 128), (25, 50)),
    ],
)
def test_head_forward(shape_hidden, shape_embeddings, expected_shape):
    hidden_states = torch.rand(shape_hidden)
    item_embeddings = torch.rand(shape_embeddings)

    head = EmbeddingTyingHead()
    scores = head(hidden_states, item_embeddings)

    assert scores.shape == expected_shape
