import copy

import pytest
import torch

from replay.nn.loss import CatalogCachedGroupedCESampled, GroupedCESampled
from replay.nn.sequential.twotower import TwoTower


class _ItemTower(torch.nn.Module):
    def __init__(self, cardinality: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(cardinality, embedding_dim))
        self.forward_calls = 0

    def forward(self, candidates: torch.LongTensor | None = None) -> torch.Tensor:
        self.forward_calls += 1
        if candidates is None:
            return self.weight
        return self.weight[candidates]


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
    negatives = torch.stack([torch.randperm(20, generator=generator)[:7] for _ in range(2)])
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
    grouped_model = _make_model(GroupedCESampled(logical_batch_size=2, groups_per_batch=2))
    short_batch = _batch(1, batch_size=1)
    actual = _loss(grouped_model, short_batch)

    embeddings, positives, negatives, _, target_mask = short_batch
    active_embeddings = embeddings[target_mask.squeeze(-1)]
    active_positives = positives.squeeze(-1)[target_mask.squeeze(-1)]
    positive_logits = grouped_model.get_logits(active_embeddings, active_positives.unsqueeze(-1))
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


def test_catalog_cache_matches_uncached_loss_and_gradients():
    uncached_model = _make_model(GroupedCESampled(logical_batch_size=2, groups_per_batch=2))
    cached_model = _make_model(
        CatalogCachedGroupedCESampled(
            cardinality=20,
            logical_batch_size=2,
            groups_per_batch=2,
            accumulation_steps=3,
        )
    )
    cached_model.load_state_dict(copy.deepcopy(uncached_model.state_dict()))
    uncached_batches = [_batch(seed) for seed in range(3)]
    cached_batches = [
        tuple(tensor.detach().clone().requires_grad_(tensor.requires_grad) for tensor in batch)
        for batch in uncached_batches
    ]

    uncached_losses = []
    for batch in uncached_batches:
        loss = _loss(uncached_model, batch)
        uncached_losses.append(loss.detach())
        loss.backward()

    cached_losses = []
    for index, batch in enumerate(cached_batches):
        is_window_end = index == 2
        cached_model.loss.prepare_accumulation_microbatch(
            is_window_end=is_window_end,
            trainer_accumulation_steps=3,
            global_step=0,
        )
        loss = _loss(cached_model, batch)
        cached_losses.append(loss.detach())
        loss.backward(retain_graph=cached_model.loss.should_retain_graph_for_current_backward())
        cached_model.loss.assert_current_backward_completed()

    cached_model.loss.assert_accumulation_cache_idle()
    torch.testing.assert_close(torch.stack(cached_losses), torch.stack(uncached_losses))
    torch.testing.assert_close(
        cached_model.body.item_tower.weight.grad,
        uncached_model.body.item_tower.weight.grad,
    )
    for cached_batch, uncached_batch in zip(cached_batches, uncached_batches, strict=True):
        torch.testing.assert_close(cached_batch[0].grad, uncached_batch[0].grad)
    assert cached_model.body.item_tower.forward_calls == 1
    assert uncached_model.body.item_tower.forward_calls == 12
    assert cached_model.state_dict().keys() == uncached_model.state_dict().keys()


def test_catalog_cache_rejects_stochastic_item_tower():
    loss = CatalogCachedGroupedCESampled(
        cardinality=20,
        logical_batch_size=2,
        groups_per_batch=2,
        accumulation_steps=2,
    )
    model = _make_model(loss)
    model.body.item_tower.dropout = torch.nn.Dropout()
    model.loss.prepare_accumulation_microbatch(
        is_window_end=False,
        trainer_accumulation_steps=2,
        global_step=0,
    )

    with pytest.raises(TypeError, match="deterministic"):
        _loss(model, _batch(0))


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
    generic_model = _make_model(GroupedCESampled(logical_batch_size=2, groups_per_batch=2))
    fast_model = _make_model(GroupedCESampled(logical_batch_size=2, groups_per_batch=2, cardinality=20))
    fast_model.load_state_dict(copy.deepcopy(generic_model.state_dict()))
    batch = _batch(3)

    generic_loss = _loss(generic_model, batch)
    fast_loss = _loss(fast_model, batch)

    torch.testing.assert_close(fast_loss, generic_loss)
