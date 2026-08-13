import copy

import pytest
import torch

from replay.nn.loss import GroupedCESampled
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
    negatives = torch.stack(
        [torch.randperm(20, generator=generator)[:7] for _ in range(2)]
    )
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


def test_grouped_loss_preserves_short_final_batch_weighting():
    grouped_model = _make_model(
        GroupedCESampled(logical_batch_size=2, groups_per_batch=2)
    )
    short_batch = _batch(1, batch_size=1)
    actual = _loss(grouped_model, short_batch)

    embeddings, positives, negatives, _, target_mask = short_batch
    active_embeddings = embeddings[target_mask.squeeze(-1)]
    active_positives = positives.squeeze(-1)[target_mask.squeeze(-1)]
    positive_logits = grouped_model.get_logits(
        active_embeddings, active_positives.unsqueeze(-1)
    )
    negative_logits = grouped_model.get_logits(active_embeddings, negatives[0])
    negative_logits[active_positives.unsqueeze(-1) == negatives[0]] = -1e9
    reference = (
        torch.nn.functional.cross_entropy(
            torch.cat((positive_logits, negative_logits), dim=-1),
            torch.zeros(active_embeddings.size(0), dtype=torch.long),
        )
        / 2
    )

    torch.testing.assert_close(actual, reference)


def test_grouped_loss_validates_negative_pool_size():
    model = _make_model(
        GroupedCESampled(
            logical_batch_size=2,
            groups_per_batch=2,
            expected_num_negatives=8,
        )
    )

    with pytest.raises(ValueError, match="must contain 8"):
        _loss(model, _batch(0))


def test_grouped_loss_ignores_padded_negatives_before_item_lookup():
    model = _make_model(
        GroupedCESampled(
            logical_batch_size=2,
            groups_per_batch=2,
            negative_labels_ignore_index=-100,
        )
    )
    batch = list(_batch(0))
    batch[2][:, -1] = -100

    loss = _loss(model, tuple(batch))

    assert torch.isfinite(loss)


def test_grouped_loss_fast_collision_mask_matches_generic_path():
    generic_model = _make_model(
        GroupedCESampled(logical_batch_size=2, groups_per_batch=2)
    )
    fast_model = _make_model(
        GroupedCESampled(logical_batch_size=2, groups_per_batch=2, cardinality=20)
    )
    fast_model.load_state_dict(copy.deepcopy(generic_model.state_dict()))
    batch = _batch(3)

    generic_loss = _loss(generic_model, batch)
    fast_loss = _loss(fast_model, batch)

    torch.testing.assert_close(fast_loss, generic_loss)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"logical_batch_size": 0, "groups_per_batch": 2}, "logical_batch_size"),
        ({"logical_batch_size": 2, "groups_per_batch": 1}, "groups_per_batch"),
        (
            {
                "logical_batch_size": 2,
                "groups_per_batch": 2,
                "expected_num_negatives": 0,
            },
            "expected_num_negatives",
        ),
        (
            {"logical_batch_size": 2, "groups_per_batch": 2, "cardinality": 0},
            "cardinality",
        ),
        (
            {"logical_batch_size": 2, "groups_per_batch": 2, "reduction": "sum"},
            "mean reduction",
        ),
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
            "exceeds",
        ),
    ],
)
def test_grouped_loss_validates_input_shapes(
    model_embeddings, positive_labels, negative_labels, target_mask, message
):
    with pytest.raises(ValueError, match=message):
        GroupedCESampled(logical_batch_size=2, groups_per_batch=2)._validate_inputs(
            model_embeddings,
            positive_labels,
            negative_labels,
            target_mask,
        )


def test_grouped_loss_rejects_empty_logical_group():
    model = _make_model(GroupedCESampled(logical_batch_size=2, groups_per_batch=2))
    batch = list(_batch(0))
    batch[-1][:2] = False

    with pytest.raises(ValueError, match="at least one target"):
        _loss(model, tuple(batch))
