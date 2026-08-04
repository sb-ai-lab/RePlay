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


@pytest.mark.parametrize(
    "shape_hidden, shape_embeddings",
    [
        ((4, 3, 8), (100, 8)),
        ((4, 8), (4, 100, 8)),
    ],
)
def test_head_matches_transposed_matmul(shape_hidden, shape_embeddings):
    hidden_states = torch.randn(shape_hidden, requires_grad=True)
    item_embeddings = torch.randn(shape_embeddings, requires_grad=True)
    reference_hidden_states = hidden_states.detach().clone().requires_grad_()
    reference_item_embeddings = item_embeddings.detach().clone().requires_grad_()

    actual = EmbeddingTyingHead()(hidden_states, item_embeddings)
    transposed_embeddings = reference_item_embeddings.transpose(-1, -2).contiguous()
    if reference_item_embeddings.dim() == 3:
        reference = reference_hidden_states.unsqueeze(-2).matmul(transposed_embeddings).squeeze(-2)
    else:
        reference = reference_hidden_states.matmul(transposed_embeddings)
    actual.sum().backward()
    reference.sum().backward()

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(hidden_states.grad, reference_hidden_states.grad)
    torch.testing.assert_close(item_embeddings.grad, reference_item_embeddings.grad)
