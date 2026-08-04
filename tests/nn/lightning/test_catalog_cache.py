from functools import partial
from types import SimpleNamespace

import lightning
import pytest
import torch
from torch.utils.data import DataLoader

from replay.nn.lightning import CatalogCacheLightningModule
from replay.nn.lightning.optimizer import OptimizerFactory
from replay.nn.loss import CatalogCachedGroupedCESampled
from replay.nn.sequential.twotower import TwoTower


class _ItemTower(torch.nn.Module):
    def __init__(self, cardinality: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(cardinality, embedding_dim))
        self.forward_calls = 0

    def forward(self, candidates: torch.LongTensor | None = None) -> torch.Tensor:
        self.forward_calls += 1
        return self.weight if candidates is None else self.weight[candidates]


class _QueryTower(torch.nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, feature_tensors, padding_mask):
        return self.projection(feature_tensors["query"])


class _Body(torch.nn.Module):
    def __init__(self, cardinality: int, embedding_dim: int) -> None:
        super().__init__()
        self.query_tower = _QueryTower(embedding_dim)
        self.item_tower = _ItemTower(cardinality, embedding_dim)

    def reset_parameters(self) -> None:
        pass


def _catalog_rows(cardinality: int, embedding_dim: int) -> list[dict]:
    return [
        {
            "feature_tensors": {"query": torch.randn(3, embedding_dim)},
            "padding_mask": torch.ones(3, dtype=torch.bool),
            "positive_labels": torch.randint(0, cardinality, (3, 1)),
            "target_padding_mask": torch.ones(3, 1, dtype=torch.bool),
        }
        for _ in range(10)
    ]


def _catalog_collate(batch: list[dict], cardinality: int) -> dict:
    result = torch.utils.data.default_collate(batch)
    result["negative_labels"] = torch.stack([torch.randperm(cardinality)[:7] for _ in range(2)])
    return result


def test_accumulation_window_end_includes_short_final_window():
    module = CatalogCacheLightningModule(torch.nn.Linear(2, 2))
    module._trainer = SimpleNamespace(accumulate_grad_batches=3, num_training_batches=5)

    assert not module._is_accumulation_window_end(0)
    assert not module._is_accumulation_window_end(1)
    assert module._is_accumulation_window_end(2)
    assert not module._is_accumulation_window_end(3)
    assert module._is_accumulation_window_end(4)


def test_catalog_cache_module_rejects_model_without_cache_loss():
    module = CatalogCacheLightningModule(torch.nn.Linear(2, 2))

    with pytest.raises(TypeError, match="missing methods"):
        module._catalog_cache_loss()


def test_catalog_cache_runs_complete_lightning_accumulation_window():
    cardinality, embedding_dim = 20, 8
    loss = CatalogCachedGroupedCESampled(
        cardinality=cardinality,
        logical_batch_size=2,
        groups_per_batch=2,
        expected_num_negatives=7,
        accumulation_steps=3,
    )
    model = TwoTower(_Body(cardinality, embedding_dim), loss)
    module = CatalogCacheLightningModule(
        model,
        optimizer_factory=OptimizerFactory(optimizer="sgd", learning_rate=0.01),
    )
    trainer = lightning.Trainer(
        max_epochs=1,
        accumulate_grad_batches=3,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(
        module,
        train_dataloaders=DataLoader(
            _catalog_rows(cardinality, embedding_dim),
            batch_size=4,
            collate_fn=partial(_catalog_collate, cardinality=cardinality),
        ),
    )

    model.loss.assert_accumulation_cache_idle()
    assert model.body.item_tower.forward_calls == 1


def test_catalog_cache_runs_with_two_cpu_ddp_processes():
    cardinality, embedding_dim = 20, 8
    loss = CatalogCachedGroupedCESampled(
        cardinality=cardinality,
        logical_batch_size=2,
        groups_per_batch=2,
        expected_num_negatives=7,
        accumulation_steps=3,
    )
    module = CatalogCacheLightningModule(
        TwoTower(_Body(cardinality, embedding_dim), loss),
        optimizer_factory=OptimizerFactory(optimizer="sgd", learning_rate=0.01),
    )
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp_spawn",
        max_epochs=1,
        accumulate_grad_batches=3,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(
        module,
        train_dataloaders=DataLoader(
            _catalog_rows(cardinality, embedding_dim),
            batch_size=4,
            collate_fn=partial(_catalog_collate, cardinality=cardinality),
        ),
    )
