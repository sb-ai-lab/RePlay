import copy

import pytest
import torch

from replay.nn.loss import CESampled


def _sampled_batch(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(2, 3, 14, generator=generator, requires_grad=True)
    positives = torch.randint(0, 15, (2, 3, 1), generator=generator)
    negatives = torch.randperm(15, generator=generator)[:7]
    target_mask = torch.ones_like(positives, dtype=torch.bool)
    return embeddings, positives, negatives, target_mask


def _loss(model, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    embeddings, positives, negatives, target_mask = batch
    return model.loss(
        model_embeddings=embeddings,
        feature_tensors={},
        positive_labels=positives,
        negative_labels=negatives,
        padding_mask=torch.ones(embeddings.shape[:2], dtype=torch.bool),
        target_padding_mask=target_mask,
    )


def _set_sampled_loss(model) -> None:
    model.loss = CESampled()
    model.loss.logits_callback = model.get_logits


def test_item_training_cache_matches_uncached_sampled_loss_and_gradients(
    create_twotower_model,
):
    uncached = create_twotower_model()
    cached = create_twotower_model()
    _set_sampled_loss(uncached)
    _set_sampled_loss(cached)
    cached.load_state_dict(copy.deepcopy(uncached.state_dict()))
    cached.body.item_tower.training_cache = True
    uncached.train()
    cached.train()

    uncached_batches = [_sampled_batch(seed) for seed in range(3)]
    cached_batches = [
        tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in batch)
        for batch in uncached_batches
    ]
    cached_calls = 0
    original_encode = cached.body.item_tower._encode_item_rows

    def count_catalog_builds(*args, **kwargs):
        nonlocal cached_calls
        cached_calls += 1
        return original_encode(*args, **kwargs)

    cached.body.item_tower._encode_item_rows = count_catalog_builds
    training_cache = cached.get_training_cache()

    uncached_losses = []
    for batch in uncached_batches:
        loss = _loss(uncached, batch)
        uncached_losses.append(loss.detach())
        loss.backward()

    cached_losses = []
    for index, batch in enumerate(cached_batches):
        training_cache.prepare_training_cache(is_window_end=index == len(cached_batches) - 1)
        loss = _loss(cached, batch)
        cached_losses.append(loss.detach())
        loss.backward(retain_graph=training_cache.should_retain_training_cache_graph())
        training_cache.assert_training_cache_backward_completed()

    training_cache.assert_training_cache_idle()
    torch.testing.assert_close(torch.stack(cached_losses), torch.stack(uncached_losses))
    for (_, cached_parameter), (_, uncached_parameter) in zip(
        cached.body.item_tower.named_parameters(),
        uncached.body.item_tower.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(cached_parameter.grad, uncached_parameter.grad)
    for cached_batch, uncached_batch in zip(cached_batches, uncached_batches, strict=True):
        torch.testing.assert_close(cached_batch[0].grad, uncached_batch[0].grad)
    assert cached_calls == 1
    assert cached.state_dict().keys() == uncached.state_dict().keys()


def test_item_training_cache_requires_a_prepared_microbatch(create_twotower_model):
    model = create_twotower_model()
    _set_sampled_loss(model)
    model.body.item_tower.training_cache = True
    model.train()

    with pytest.raises(RuntimeError, match="prepare_training_cache"):
        _loss(model, _sampled_batch(0))


def test_item_training_cache_rejects_stochastic_modules(create_twotower_model):
    model = create_twotower_model()
    _set_sampled_loss(model)
    model.body.item_tower.training_cache = True
    model.body.item_tower.dropout = torch.nn.Dropout()
    model.train()
    model.get_training_cache().prepare_training_cache(is_window_end=True)

    with pytest.raises(TypeError, match="deterministic"):
        _loss(model, _sampled_batch(0))
