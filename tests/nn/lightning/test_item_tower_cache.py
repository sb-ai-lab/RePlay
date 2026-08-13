from types import SimpleNamespace

import lightning
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from replay.nn.lightning import ItemTowerCacheLightningModule
from replay.nn.lightning.optimizer import OptimizerFactory
from replay.nn.sequential.twotower import TwoTower


class _CacheController:
    def __init__(self) -> None:
        self.idle_checks = 0

    def prepare_training_cache(self, is_window_end: bool) -> None:
        pass

    def should_retain_training_cache_graph(self) -> bool:
        return False

    def assert_training_cache_backward_completed(self) -> None:
        pass

    def assert_training_cache_idle(self) -> None:
        self.idle_checks += 1


class _CacheModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cache_controller = _CacheController()

    def get_training_cache(self) -> _CacheController:
        return self.cache_controller


class _FixedBatchDataset(Dataset):
    def __init__(self, batch: dict, num_batches: int) -> None:
        self.batch = batch
        self.num_batches = num_batches

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= self.num_batches:
            raise IndexError(index)
        return self.batch


class _FixedBatchIterableDataset(IterableDataset):
    def __init__(self, batch: dict, num_batches: int) -> None:
        self.batch = batch
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.batch


def _make_model(schema, item_features_reader) -> TwoTower:
    return TwoTower.from_params(
        schema=schema,
        item_features_reader=item_features_reader,
        embedding_dim=schema["item_id"].embedding_dim,
        num_heads=1,
        num_blocks=1,
        max_sequence_length=7,
        dropout=0.2,
        item_training_cache=True,
    )


def _training_dataloader(simple_batch: dict, iterable: bool = False, num_batches: int = 10) -> DataLoader:
    item_ids = simple_batch["feature_tensors"]["item_id"]
    padding_mask = simple_batch["padding_mask"]
    batch = {
        "feature_tensors": simple_batch["feature_tensors"],
        "padding_mask": padding_mask,
        "positive_labels": item_ids.unsqueeze(-1),
        "negative_labels": torch.zeros(1, dtype=torch.long),
        "target_padding_mask": padding_mask.unsqueeze(-1),
    }
    dataset_class = _FixedBatchIterableDataset if iterable else _FixedBatchDataset
    return DataLoader(dataset_class(batch, num_batches=num_batches), batch_size=None)


def test_accumulation_window_end_uses_lightning_decision():
    module = ItemTowerCacheLightningModule(torch.nn.Linear(2, 2))
    module._trainer = SimpleNamespace(fit_loop=SimpleNamespace(_should_accumulate=lambda: True))

    assert not module._is_accumulation_window_end()
    module.trainer.fit_loop._should_accumulate = lambda: False
    assert module._is_accumulation_window_end()


def test_item_tower_cache_module_rejects_model_without_cache_interface():
    module = ItemTowerCacheLightningModule(torch.nn.Linear(2, 2))

    with pytest.raises(TypeError, match="get_training_cache"):
        module._training_cache()


def test_item_tower_cache_module_validates_lightning_lifecycle():
    model = _CacheModel()
    module = ItemTowerCacheLightningModule(model)
    module._trainer = SimpleNamespace(
        accumulate_grad_batches=2,
        num_training_batches=2,
        strategy=SimpleNamespace(handles_gradient_accumulation=False),
    )

    with pytest.raises(ValueError, match="controls retain_graph"):
        module.backward(torch.tensor(1.0, requires_grad=True), retain_graph=True)

    module.on_validation_epoch_start()
    module.on_save_checkpoint({})
    assert model.cache_controller.idle_checks == 2

    module.automatic_optimization = False
    with pytest.raises(RuntimeError, match="automatic optimization"):
        module.on_train_start()

    module.automatic_optimization = True
    module.trainer.strategy.handles_gradient_accumulation = True
    with pytest.raises(RuntimeError, match="handles gradient accumulation"):
        module.on_train_start()


def test_item_tower_cache_runs_with_standard_loss(
    tensor_schema_with_equal_embedding_dims,
    item_features_reader,
    simple_batch,
):
    model = _make_model(tensor_schema_with_equal_embedding_dims, item_features_reader)
    module = ItemTowerCacheLightningModule(
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

    trainer.fit(module, train_dataloaders=_training_dataloader(simple_batch))

    model.get_training_cache().assert_training_cache_idle()


def test_item_tower_cache_handles_short_iterable_accumulation_window(
    tensor_schema_with_equal_embedding_dims,
    item_features_reader,
    simple_batch,
):
    model = _make_model(tensor_schema_with_equal_embedding_dims, item_features_reader)
    module = ItemTowerCacheLightningModule(model)
    trainer = lightning.Trainer(
        max_epochs=1,
        accumulate_grad_batches=3,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(module, train_dataloaders=_training_dataloader(simple_batch, iterable=True, num_batches=5))

    model.get_training_cache().assert_training_cache_idle()


def test_item_tower_cache_runs_with_two_cpu_ddp_processes(
    tensor_schema_with_equal_embedding_dims,
    item_features_reader,
    simple_batch,
):
    model = _make_model(tensor_schema_with_equal_embedding_dims, item_features_reader)
    module = ItemTowerCacheLightningModule(
        model,
        optimizer_factory=OptimizerFactory(optimizer="sgd", learning_rate=0.01),
    )
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp_fork",
        max_epochs=1,
        accumulate_grad_batches=3,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(module, train_dataloaders=_training_dataloader(simple_batch))
