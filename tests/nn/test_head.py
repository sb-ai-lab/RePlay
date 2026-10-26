import pytest
import torch

from replay.nn.head import ClassificationHead, EmbeddingTyingHead


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
    hidden_states = torch.randn(*shape_hidden[:-1], shape_hidden[-1] * 2)[..., ::2].requires_grad_()
    item_embeddings = torch.randn(*shape_embeddings[:-1], shape_embeddings[-1] * 2)[..., ::2].requires_grad_()
    reference_hidden_states = hidden_states.detach().clone().requires_grad_()
    reference_item_embeddings = item_embeddings.detach().clone().requires_grad_()

    assert not hidden_states.is_contiguous()
    assert not item_embeddings.is_contiguous()

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


@pytest.mark.parametrize(
    "shape_hidden, expected_shape",
    [
        ((25, 128), (25, 10)),
        ((25, 50, 128), (25, 50, 10)),
    ],
)
def test_classification_head_forward(shape_hidden, expected_shape):
    hidden_states = torch.rand(shape_hidden)
    item_embeddings = torch.rand(10, 128)

    head = ClassificationHead(hidden_size=128, n_classes=10)
    scores = head(hidden_states, item_embeddings)

    assert scores.shape == expected_shape


def test_classification_head_reset_parameters():
    head = ClassificationHead(hidden_size=3, n_classes=2)
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.fill_(3.0)

    head.reset_parameters()

    assert torch.count_nonzero(head.linear.weight) > 0
    assert (head.linear.bias != 3.0).all()
