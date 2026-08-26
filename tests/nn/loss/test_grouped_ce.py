import copy

import pytest
import torch

from replay.nn.loss import CESampled, GroupedCESampled
from replay.nn.loss._grouped import evaluate_group_losses
from replay.nn.loss.base import mask_negative_logits
from replay.nn.sequential.twotower import TwoTower


class _ItemTower(torch.nn.Module):
    def __init__(self, cardinality: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(cardinality, embedding_dim))

    def forward(self, candidates: torch.LongTensor | None = None) -> torch.Tensor:
        return self.weight if candidates is None else self.weight[candidates]


class _Body(torch.nn.Module):
    def __init__(self, item_tower: torch.nn.Module) -> None:
        super().__init__()
        self.item_tower = item_tower

    def reset_parameters(self) -> None:
        pass


def _make_model(loss: torch.nn.Module) -> TwoTower:
    return TwoTower(body=_Body(_ItemTower(cardinality=20, embedding_dim=8)), loss=loss)


def _batch(seed: int, batch_size: int = 4) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(batch_size, 3, 8, generator=generator, requires_grad=True)
    positives = torch.randint(0, 20, (batch_size, 3, 1), generator=generator)
    num_groups = (batch_size + 1) // 2
    negatives = torch.stack([torch.randperm(20, generator=generator)[:7] for _ in range(num_groups)])
    target_mask = torch.ones_like(positives, dtype=torch.bool)
    padding_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    return embeddings, positives, negatives, padding_mask, target_mask


def _loss(model: TwoTower, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    embeddings, positives, negatives, padding_mask, target_mask = batch
    return model.loss(
        model_embeddings=embeddings,
        feature_tensors={},
        positive_labels=positives,
        negative_labels=negatives,
        padding_mask=padding_mask,
        target_padding_mask=target_mask,
    )


def _assert_matches_logical_batches(
    batch: tuple[torch.Tensor, ...],
    reference_loss: CESampled,
    grouped_loss: GroupedCESampled,
) -> None:
    embeddings, positives, negatives, padding_mask, target_mask = batch
    expected_embeddings = embeddings.detach().clone().requires_grad_(True)
    actual_embeddings = embeddings.detach().clone().requires_grad_(True)
    reference_model = _make_model(reference_loss)
    grouped_model = _make_model(grouped_loss)
    grouped_model.load_state_dict(copy.deepcopy(reference_model.state_dict()))

    logical_batch_size = grouped_loss.logical_batch_size
    expected = torch.stack(
        [
            reference_model.loss(
                model_embeddings=expected_embeddings[start : start + logical_batch_size],
                feature_tensors={},
                positive_labels=positives[start : start + logical_batch_size],
                negative_labels=negatives[group],
                padding_mask=padding_mask[start : start + logical_batch_size],
                target_padding_mask=target_mask[start : start + logical_batch_size],
            )
            for group, start in enumerate(range(0, expected_embeddings.size(0), logical_batch_size))
        ]
    ).mean()
    actual = grouped_model.loss(
        model_embeddings=actual_embeddings,
        feature_tensors={},
        positive_labels=positives,
        negative_labels=negatives,
        padding_mask=padding_mask,
        target_padding_mask=target_mask,
    )

    torch.testing.assert_close(actual, expected)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual_embeddings.grad, expected_embeddings.grad)
    torch.testing.assert_close(
        grouped_model.body.item_tower.weight.grad,
        reference_model.body.item_tower.weight.grad,
    )


def test_grouped_loss_averages_active_logical_groups():
    grouped_model = _make_model(GroupedCESampled(logical_batch_size=2))
    short_batch = _batch(1, batch_size=1)
    actual = _loss(grouped_model, short_batch)

    embeddings, positives, negatives, _, target_mask = short_batch
    active_embeddings = embeddings[target_mask.squeeze(-1)]
    active_positives = positives.squeeze(-1)[target_mask.squeeze(-1)]
    positive_logits = grouped_model.get_logits(active_embeddings, active_positives.unsqueeze(-1))
    negative_logits = grouped_model.get_logits(active_embeddings, negatives[0])
    negative_logits[active_positives.unsqueeze(-1) == negatives[0]] = -1e9
    reference = torch.nn.functional.cross_entropy(
        torch.cat((positive_logits, negative_logits), dim=-1),
        torch.zeros(active_embeddings.size(0), dtype=torch.long),
    )

    torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("cardinality", [None, 20])
def test_grouped_loss_matches_logical_batches_and_gradients(cardinality):
    batch = list(_batch(17, batch_size=5))
    batch[2][:, -1] = -100
    batch[-1][0, 0] = False
    batch[-1][1, 1:] = False
    batch[-1][2, :2] = False
    batch[-1][4, 0] = False
    _assert_matches_logical_batches(
        tuple(batch),
        reference_loss=CESampled(negative_labels_ignore_index=-100, label_smoothing=0.1),
        grouped_loss=GroupedCESampled(
            logical_batch_size=2,
            cardinality=cardinality,
            negative_labels_ignore_index=-100,
            label_smoothing=0.1,
        ),
    )


@pytest.mark.parametrize("cardinality", [None, 20])
def test_grouped_loss_matches_duplicate_negative_pools(cardinality):
    batch = list(_batch(19, batch_size=4))
    batch[2][0, 1] = batch[2][0, 0]
    batch[2][1, 2] = batch[2][1, 0]
    _assert_matches_logical_batches(
        tuple(batch),
        reference_loss=CESampled(),
        grouped_loss=GroupedCESampled(logical_batch_size=2, cardinality=cardinality),
    )


def test_grouped_loss_ignores_padded_negatives_before_item_lookup():
    model = _make_model(
        GroupedCESampled(
            logical_batch_size=2,
            negative_labels_ignore_index=-100,
        )
    )
    batch = list(_batch(0))
    batch[2][:, -1] = -100

    loss = _loss(model, tuple(batch))

    assert torch.isfinite(loss)


def test_grouped_loss_supports_empty_negative_pools():
    model = _make_model(GroupedCESampled(logical_batch_size=2, cardinality=20))
    batch = list(_batch(0))
    batch[2] = torch.empty((2, 0), dtype=torch.long)

    loss = _loss(model, tuple(batch))

    torch.testing.assert_close(loss, torch.zeros_like(loss))


def test_grouped_loss_fast_collision_mask_matches_generic_path():
    generic_model = _make_model(GroupedCESampled(logical_batch_size=2, negative_labels_ignore_index=-100))
    fast_model = _make_model(
        GroupedCESampled(
            logical_batch_size=2,
            cardinality=20,
            negative_labels_ignore_index=-100,
        )
    )
    fast_model.load_state_dict(copy.deepcopy(generic_model.state_dict()))
    batch = list(_batch(3))
    batch[1][0, 0, 0] = batch[2][0, 0]
    batch[2][:, -1] = -100

    generic_loss = _loss(generic_model, tuple(batch))
    fast_loss = _loss(fast_model, tuple(batch))

    torch.testing.assert_close(fast_loss, generic_loss)


def test_grouped_loss_uses_dense_collision_lookup_only_for_single_group():
    single_group_model = _make_model(GroupedCESampled(logical_batch_size=2, cardinality=20))
    packed_model = _make_model(GroupedCESampled(logical_batch_size=2, cardinality=20))

    _loss(single_group_model, _batch(1, batch_size=1))
    _loss(packed_model, _batch(2, batch_size=4))

    assert single_group_model.loss._negative_column_lookup is not None
    assert packed_model.loss._negative_column_lookup is None


@pytest.mark.parametrize("cardinality", [None, 20])
def test_single_group_duplicate_pool_keeps_exact_fallback(cardinality):
    batch = list(_batch(8, batch_size=1))
    batch[2][0, 1] = batch[2][0, 0]
    _assert_matches_logical_batches(
        tuple(batch),
        reference_loss=CESampled(),
        grouped_loss=GroupedCESampled(logical_batch_size=2, cardinality=cardinality),
    )


def test_grouped_loss_falls_back_when_vmap_is_unavailable(monkeypatch):
    expected_input = torch.randn(3, 4, requires_grad=True)
    actual_input = expected_input.detach().clone().requires_grad_(True)

    def group_loss(values: torch.Tensor) -> torch.Tensor:
        return values.square().mean()

    expected = torch.stack([group_loss(values) for values in expected_input])
    monkeypatch.setattr(torch, "vmap", None, raising=False)
    actual = evaluate_group_losses(group_loss, actual_input)

    torch.testing.assert_close(actual, expected)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad)


def test_shared_negative_lookup_requires_one_shared_pool():
    with pytest.raises(ValueError, match="one shared pool"):
        mask_negative_logits(
            torch.zeros(2, 3),
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, 1, dtype=torch.long),
            -100,
            negative_column_lookup=torch.empty(20, dtype=torch.int32),
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"logical_batch_size": 0}, "logical_batch_size"),
        ({"logical_batch_size": 2, "cardinality": 0}, "cardinality"),
        ({"logical_batch_size": 2, "reduction": "sum"}, "mean reduction"),
    ],
)
def test_grouped_loss_validates_constructor_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GroupedCESampled(**kwargs)


@pytest.mark.parametrize(
    "model_embeddings, positive_labels, negative_labels, target_mask, message",
    [
        (
            torch.zeros(2, 3),
            torch.zeros(2, 3, 1, dtype=torch.long),
            torch.zeros(2, 4, dtype=torch.long),
            torch.ones(2, 3, 1, dtype=torch.bool),
            "model_embeddings",
        ),
        (
            torch.zeros(2, 3, 1),
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, 4, dtype=torch.long),
            torch.ones(2, 3, 1, dtype=torch.bool),
            "one positive",
        ),
        (
            torch.zeros(2, 3, 1),
            torch.zeros(1, 3, 1, dtype=torch.long),
            torch.zeros(2, 4, dtype=torch.long),
            torch.ones(1, 3, 1, dtype=torch.bool),
            "equal batch",
        ),
        (
            torch.zeros(2, 3, 1),
            torch.zeros(2, 3, 1, dtype=torch.long),
            torch.zeros(2, 4, dtype=torch.long),
            torch.ones(2, 3, dtype=torch.bool),
            "target_padding_mask",
        ),
        (
            torch.zeros(2, 3, 1),
            torch.zeros(2, 3, 1, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            torch.ones(2, 3, 1, dtype=torch.bool),
            "negative_labels",
        ),
        (
            torch.zeros(0, 3, 1),
            torch.zeros(0, 3, 1, dtype=torch.long),
            torch.zeros(2, 4, dtype=torch.long),
            torch.ones(0, 3, 1, dtype=torch.bool),
            "empty batches",
        ),
        (
            torch.zeros(5, 3, 1),
            torch.zeros(5, 3, 1, dtype=torch.long),
            torch.ones(2, 4, dtype=torch.long),
            torch.ones(5, 3, 1, dtype=torch.bool),
            "must contain 3 pools",
        ),
    ],
)
def test_grouped_loss_validates_input_shapes(model_embeddings, positive_labels, negative_labels, target_mask, message):
    model = _make_model(GroupedCESampled(logical_batch_size=2))
    with pytest.raises(ValueError, match=message):
        _loss(
            model,
            (
                model_embeddings,
                positive_labels,
                negative_labels,
                torch.empty(0, dtype=torch.bool),
                target_mask,
            ),
        )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_grouped_loss_rejects_empty_logical_group(batch_size):
    model = _make_model(GroupedCESampled(logical_batch_size=2))
    batch = list(_batch(0, batch_size=batch_size))
    batch[-1][:2] = False

    with pytest.raises(ValueError, match="at least one target"):
        _loss(model, tuple(batch))
