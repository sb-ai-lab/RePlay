import pytest
import torch

from replay.nn.loss import GroupedLogInCESampled, LogInCESampled
from replay.nn.sequential.twotower import TwoTower
from replay.nn.transform import GroupedUniformNegativeSamplingTransform


class _ItemTower(torch.nn.Module):
    def __init__(self, cardinality: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(cardinality, embedding_dim))

    def forward(self, candidates: torch.LongTensor | None = None) -> torch.Tensor:
        return self.weight if candidates is None else self.weight[candidates]


class _QueryTower(torch.nn.Module):
    def forward(self, feature_tensors: dict[str, torch.Tensor], padding_mask: torch.BoolTensor) -> torch.Tensor:
        return feature_tensors["query_embeddings"] * padding_mask.unsqueeze(-1)


class _Body(torch.nn.Module):
    def __init__(self, cardinality: int, embedding_dim: int) -> None:
        super().__init__()
        self.query_tower = _QueryTower()
        self.item_tower = _ItemTower(cardinality, embedding_dim)

    def reset_parameters(self) -> None:
        pass


def _logits_callback(query: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    item_embeddings = torch.arange(48, dtype=query.dtype).reshape(12, 4) / 10
    if labels.ndim == 1:
        return query @ item_embeddings[labels].T
    return torch.einsum("bd,bnd->bn", query, item_embeddings[labels])


def _batch(batch_size: int = 5) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(11)
    embeddings = torch.randn(batch_size, 3, 4, generator=generator)
    positives = torch.randint(0, 12, (batch_size, 3, 2), generator=generator)
    negatives = torch.stack([torch.randperm(12, generator=generator)[:5] for _ in range((batch_size + 1) // 2)])
    padding_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    target_mask = torch.ones_like(positives, dtype=torch.bool)
    return embeddings, positives, negatives, padding_mask, target_mask


def _loss(loss, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    embeddings, positives, negatives, padding_mask, target_mask = batch
    return loss(embeddings, {}, positives, negatives, padding_mask, target_mask)


def test_grouped_login_ce_matches_logical_multipositive_batches():
    batch = _batch()
    expected_embeddings = batch[0].detach().clone().requires_grad_(True)
    actual_embeddings = batch[0].detach().clone().requires_grad_(True)
    _, positives, negatives, padding_mask, target_mask = batch

    reference = LogInCESampled()
    reference.logits_callback = _logits_callback
    expected = torch.stack(
        [
            reference(
                expected_embeddings[start : start + 2],
                {},
                positives[start : start + 2],
                negatives[group],
                padding_mask[start : start + 2],
                target_mask[start : start + 2],
            )
            for group, start in enumerate(range(0, expected_embeddings.size(0), 2))
        ]
    ).mean()

    grouped = GroupedLogInCESampled(logical_batch_size=2)
    grouped.logits_callback = _logits_callback
    actual = _loss(grouped, (actual_embeddings, positives, negatives, padding_mask, target_mask))

    torch.testing.assert_close(actual, expected)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual_embeddings.grad, expected_embeddings.grad)


def test_grouped_login_ce_forwards_loss_parameters():
    loss = GroupedLogInCESampled(
        logical_batch_size=2,
        log_epsilon=1e-4,
        clamp_border=17.0,
        negative_labels_ignore_index=-7,
    )

    assert loss.log_epsilon == 1e-4
    assert loss.clamp_border == 17.0
    assert loss.negative_labels_ignore_index == -7


def test_grouped_login_ce_end_to_end_with_sampler_and_twotower():
    logical_batch_size = 2
    sampler = GroupedUniformNegativeSamplingTransform(
        cardinality=12,
        num_negative_samples=5,
        group_size=logical_batch_size,
        generator=torch.Generator().manual_seed(7),
    )
    loss = GroupedLogInCESampled(logical_batch_size=logical_batch_size)
    model = TwoTower(body=_Body(cardinality=12, embedding_dim=4), loss=loss)
    embeddings, positives, _, padding_mask, target_mask = _batch(batch_size=5)
    embeddings.requires_grad_(True)
    sampled_batch = sampler({"positive_labels": positives})

    output = model.forward_train(
        feature_tensors={"query_embeddings": embeddings},
        positive_labels=positives,
        negative_labels=sampled_batch["negative_labels"],
        padding_mask=padding_mask,
        target_padding_mask=target_mask,
    )

    assert loss.logical_batch_size == sampler.group_size
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert model.body.item_tower.weight.grad is not None
    assert torch.isfinite(model.body.item_tower.weight.grad).all()


def test_grouped_login_ce_ignores_padded_negatives_before_item_lookup():
    loss = GroupedLogInCESampled(logical_batch_size=2, negative_labels_ignore_index=-100)
    loss.logits_callback = _logits_callback
    batch = list(_batch())
    batch[2][:, -1] = -100

    assert torch.isfinite(_loss(loss, tuple(batch)))


def test_grouped_login_ce_validates_constructor_arguments():
    with pytest.raises(ValueError, match="logical_batch_size"):
        GroupedLogInCESampled(logical_batch_size=0)


@pytest.mark.parametrize(
    "model_embeddings, positive_labels, negative_labels, target_mask, message",
    [
        (
            torch.zeros(2, 3),
            torch.zeros(2, 3, 2, dtype=torch.long),
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(2, 3, 2, dtype=torch.bool),
            "model_embeddings",
        ),
        (
            torch.zeros(2, 3, 4),
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(2, 3, 2, dtype=torch.bool),
            "positive_labels",
        ),
        (
            torch.zeros(2, 3, 4),
            torch.zeros(1, 3, 2, dtype=torch.long),
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(1, 3, 2, dtype=torch.bool),
            "equal batch",
        ),
        (
            torch.zeros(2, 3, 4),
            torch.zeros(2, 3, 2, dtype=torch.long),
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(2, 3, dtype=torch.bool),
            "target_padding_mask",
        ),
        (
            torch.zeros(2, 3, 4),
            torch.zeros(2, 3, 2, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            torch.ones(2, 3, 2, dtype=torch.bool),
            "negative_labels",
        ),
        (
            torch.zeros(0, 3, 4),
            torch.zeros(0, 3, 2, dtype=torch.long),
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(0, 3, 2, dtype=torch.bool),
            "empty batches",
        ),
        (
            torch.zeros(5, 3, 4),
            torch.zeros(5, 3, 2, dtype=torch.long),
            torch.zeros(2, 4, dtype=torch.long),
            torch.ones(5, 3, 2, dtype=torch.bool),
            "must contain 3 pools",
        ),
    ],
)
def test_grouped_login_ce_validates_input_shapes(
    model_embeddings,
    positive_labels,
    negative_labels,
    target_mask,
    message,
):
    with pytest.raises(ValueError, match=message):
        GroupedLogInCESampled(logical_batch_size=2)._validate_inputs(
            model_embeddings,
            positive_labels,
            negative_labels,
            target_mask,
        )


def test_grouped_login_ce_rejects_empty_logical_group():
    loss = GroupedLogInCESampled(logical_batch_size=2)
    loss.logits_callback = _logits_callback
    batch = list(_batch())
    batch[-1][:2] = False

    with pytest.raises(ValueError, match="at least one target"):
        _loss(loss, tuple(batch))
