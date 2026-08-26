import pytest
import torch

from replay.nn.loss import GroupedLogInCESampled, LogInCESampled
from replay.nn.loss._grouped import (
    find_negative_collision_columns,
    mask_group_negative_logits,
)
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


def _assert_matches_logical_batches(
    batch: tuple[torch.Tensor, ...],
    reference_loss: LogInCESampled,
    grouped_loss: GroupedLogInCESampled,
) -> None:
    embeddings, positives, negatives, padding_mask, target_mask = batch
    expected_embeddings = embeddings.detach().clone().requires_grad_(True)
    actual_embeddings = embeddings.detach().clone().requires_grad_(True)
    reference_loss.logits_callback = _logits_callback
    grouped_loss.logits_callback = _logits_callback

    logical_batch_size = grouped_loss.logical_batch_size
    expected = torch.stack(
        [
            reference_loss(
                expected_embeddings[start : start + logical_batch_size],
                {},
                positives[start : start + logical_batch_size],
                negatives[group],
                padding_mask[start : start + logical_batch_size],
                target_mask[start : start + logical_batch_size],
            )
            for group, start in enumerate(range(0, expected_embeddings.size(0), logical_batch_size))
        ]
    ).mean()
    actual = _loss(
        grouped_loss,
        (actual_embeddings, positives, negatives, padding_mask, target_mask),
    )

    torch.testing.assert_close(actual, expected)
    expected.backward()
    actual.backward()
    assert torch.isfinite(actual_embeddings.grad).all()
    torch.testing.assert_close(actual_embeddings.grad, expected_embeddings.grad)


def test_grouped_login_ce_matches_logical_multipositive_batches():
    batch = list(_batch())
    batch[1][0, 0] = torch.tensor([3, 2])
    batch[2][0] = torch.tensor([1, 3, 5, 7, 9])
    _assert_matches_logical_batches(
        tuple(batch),
        reference_loss=LogInCESampled(),
        grouped_loss=GroupedLogInCESampled(logical_batch_size=2),
    )


def test_grouped_login_ce_matches_single_logical_batch():
    _assert_matches_logical_batches(
        _batch(batch_size=1),
        reference_loss=LogInCESampled(),
        grouped_loss=GroupedLogInCESampled(logical_batch_size=2),
    )


def test_grouped_login_ce_masks_a_collision_when_another_positive_is_not_sampled():
    batch = list(_batch(batch_size=1))
    batch[1][0, 0] = torch.tensor([9, 11])
    batch[2][0] = torch.tensor([1, 3, 5, 7, 9])
    _assert_matches_logical_batches(
        tuple(batch),
        reference_loss=LogInCESampled(),
        grouped_loss=GroupedLogInCESampled(logical_batch_size=2),
    )


def test_grouped_login_ce_does_not_mask_collision_from_inactive_positive():
    positive_labels = torch.tensor([[4, 7]])
    positive_labels_mask = torch.tensor([[True, False]])
    negative_labels = torch.tensor([[7, 8]])
    negative_logits = torch.zeros(1, 2)

    fallback_logits = mask_group_negative_logits(
        negative_logits,
        positive_labels,
        positive_labels_mask,
        negative_labels[0],
        -100,
    )
    sorted_labels, sorted_columns = torch.sort(negative_labels, dim=-1)
    collision_columns, collisions = find_negative_collision_columns(
        positive_labels.unsqueeze(0),
        sorted_labels,
        sorted_columns,
        positive_labels_mask.unsqueeze(0),
    )

    assert torch.isfinite(fallback_logits[0, 0])
    assert collision_columns.shape == collisions.shape == (1, 1, 2)
    assert not collisions.any()


def test_grouped_login_ce_matches_duplicate_negative_pools():
    batch = list(_batch(batch_size=4))
    batch[2][0, 1] = batch[2][0, 0]
    batch[2][1, 2] = batch[2][1, 0]
    _assert_matches_logical_batches(
        tuple(batch),
        reference_loss=LogInCESampled(),
        grouped_loss=GroupedLogInCESampled(logical_batch_size=2),
    )


def test_grouped_login_ce_matches_sparse_logical_batches():
    batch = list(_batch())
    batch[-1][0, :2] = False
    batch[1][0, 0, 0] = batch[2][0, 0]
    batch[-1][1, 1, 1] = False
    batch[-1][2, 0] = False
    batch[-1][3, 2] = False
    _assert_matches_logical_batches(
        tuple(batch),
        reference_loss=LogInCESampled(),
        grouped_loss=GroupedLogInCESampled(logical_batch_size=2),
    )


def test_grouped_login_ce_masks_padded_positions_before_logarithm():
    batch = list(_batch())
    batch[2].fill_(-100)
    _assert_matches_logical_batches(
        tuple(batch),
        reference_loss=LogInCESampled(negative_labels_ignore_index=-100),
        grouped_loss=GroupedLogInCESampled(logical_batch_size=2, negative_labels_ignore_index=-100),
    )


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
    sampled_batch = sampler(
        {
            "positive_labels": positives,
            "feature_tensors": {"query_embeddings": embeddings},
            "padding_mask": padding_mask,
            "target_padding_mask": target_mask,
        }
    )

    output = model(**sampled_batch)

    assert loss.logical_batch_size == sampler.group_size
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert model.body.item_tower.weight.grad is not None
    assert torch.isfinite(model.body.item_tower.weight.grad).all()


def test_grouped_login_ce_rejects_mismatched_sampler_group_size():
    sampler = GroupedUniformNegativeSamplingTransform(
        cardinality=12,
        num_negative_samples=5,
        group_size=4,
        generator=torch.Generator().manual_seed(3),
    )
    embeddings, positives, _, padding_mask, target_mask = _batch(batch_size=16)
    sampled_batch = sampler(
        {
            "positive_labels": positives,
            "feature_tensors": {"query_embeddings": embeddings},
            "padding_mask": padding_mask,
            "target_padding_mask": target_mask,
        }
    )
    loss = GroupedLogInCESampled(logical_batch_size=5)
    model = TwoTower(body=_Body(cardinality=12, embedding_dim=4), loss=loss)

    with pytest.raises(ValueError, match="same group size"):
        model(**sampled_batch)


def test_grouped_login_ce_ignores_padded_negatives_before_item_lookup():
    loss = GroupedLogInCESampled(logical_batch_size=2, negative_labels_ignore_index=-100)
    loss.logits_callback = _logits_callback
    batch = list(_batch())
    batch[2][:, -1] = -100

    assert torch.isfinite(_loss(loss, tuple(batch)))


def test_grouped_login_ce_supports_empty_negative_pools():
    embeddings, positives, _, padding_mask, target_mask = _batch()
    embeddings.requires_grad_(True)
    loss = GroupedLogInCESampled(logical_batch_size=2)
    model = TwoTower(body=_Body(cardinality=12, embedding_dim=4), loss=loss)

    actual = model(
        feature_tensors={"query_embeddings": embeddings},
        padding_mask=padding_mask,
        positive_labels=positives,
        negative_labels=torch.empty((3, 0), dtype=torch.long),
        target_padding_mask=target_mask,
        negative_group_size=2,
    )["loss"]

    torch.testing.assert_close(actual, torch.zeros_like(actual))
    actual.backward()
    assert embeddings.grad is not None
    torch.testing.assert_close(embeddings.grad, torch.zeros_like(embeddings.grad))
    assert model.body.item_tower.weight.grad is not None
    torch.testing.assert_close(
        model.body.item_tower.weight.grad,
        torch.zeros_like(model.body.item_tower.weight.grad),
    )


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
    loss = GroupedLogInCESampled(logical_batch_size=2)
    loss.logits_callback = _logits_callback
    with pytest.raises(ValueError, match=message):
        _loss(
            loss,
            (
                model_embeddings,
                positive_labels,
                negative_labels,
                torch.empty(0, dtype=torch.bool),
                target_mask,
            ),
        )


def test_grouped_login_ce_rejects_empty_logical_group():
    loss = GroupedLogInCESampled(logical_batch_size=2)
    loss.logits_callback = _logits_callback
    batch = list(_batch())
    batch[-1][:2] = False

    with pytest.raises(ValueError, match="at least one target"):
        _loss(loss, tuple(batch))
