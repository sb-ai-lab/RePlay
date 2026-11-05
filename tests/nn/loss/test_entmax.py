import pytest
import torch

from replay.nn.loss.entmax import _Entmax15Loss


def test_entmax_loss_validates_reduction():
    with pytest.raises(ValueError, match="reduction must"):
        _Entmax15Loss(reduction="invalid")


@pytest.mark.parametrize(
    "logits, target, error, message",
    [
        (
            torch.tensor(0.0),
            torch.tensor(0),
            ValueError,
            "floating-point logits",
        ),
        (
            torch.tensor([0, 1]),
            torch.tensor(0),
            ValueError,
            "floating-point logits",
        ),
        (
            torch.zeros(2, 3),
            torch.tensor([0.0, 1.0]),
            TypeError,
            "torch.long",
        ),
        (
            torch.zeros(2, 3),
            torch.tensor([0]),
            ValueError,
            "expected target shape",
        ),
        (
            torch.zeros(2, 0),
            torch.tensor([0, 1]),
            ValueError,
            "class axis must be nonempty",
        ),
    ],
)
def test_entmax_loss_validates_inputs(logits, target, error, message):
    with pytest.raises(error, match=message):
        _Entmax15Loss()(logits, target)


def test_entmax_loss_reductions():
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, -0.5],
            [0.0, 0.0, 0.0],
        ]
    )
    target = torch.tensor([0, 2, -100])

    unreduced = _Entmax15Loss(reduction="none")(logits, target)

    assert unreduced.shape == target.shape
    torch.testing.assert_close(unreduced[-1], torch.tensor(0.0))
    torch.testing.assert_close(
        _Entmax15Loss(reduction="sum")(logits, target),
        unreduced.sum(),
    )
    torch.testing.assert_close(
        _Entmax15Loss(reduction="mean")(logits, target),
        unreduced[:2].mean(),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_entmax_loss_promotes_low_precision_logits(dtype):
    logits = torch.tensor(
        [[0.5, 0.0, -0.5], [0.0, 0.5, -0.5]],
        dtype=dtype,
        requires_grad=True,
    )
    target = torch.tensor([0, 2])
    reference_logits = logits.detach().float().requires_grad_()

    actual = _Entmax15Loss()(logits, target)
    expected = _Entmax15Loss()(reference_logits, target)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)

    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad.to(dtype))
