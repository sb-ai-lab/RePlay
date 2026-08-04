from types import SimpleNamespace
from unittest.mock import patch

import lightning
import torch
from torch.utils.data import DataLoader

from replay.nn.lightning import LightningModule


class LossModel(torch.nn.Module):
    def __init__(self, loss: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(loss))

    def forward(self, feature_tensors):
        return {"loss": self.scale.square()}


class ConstantLossModel(LossModel):
    def forward(self, feature_tensors):
        return {"loss": self.scale * 0 + 4}


def make_batch(batch_size):
    return {
        "feature_tensors": {
            "item_id": torch.zeros(batch_size, 4, dtype=torch.long),
            "timestamp": torch.ones(batch_size, 4, dtype=torch.long),
        },
        "negative_labels": torch.arange(40_000),
    }


def test_default_training_step_keeps_existing_logging():
    module = LightningModule(LossModel(2.0))
    module._trainer = SimpleNamespace(global_rank=0)

    with patch.object(module, "optimizers", return_value=SimpleNamespace(param_groups=[{"lr": 0.01}])):
        with patch.object(module, "log") as log:
            loss = module.training_step(make_batch(2))

    assert loss.requires_grad
    assert [call.args[0] for call in log.call_args_list] == ["learning_rate", "train_loss"]
    assert all(call.kwargs["sync_dist"] is True for call in log.call_args_list)


def test_epoch_mode_defers_distributed_metrics_and_preserves_weights():
    module = LightningModule(ConstantLossModel(2.0), epoch_only_training_metrics=True)
    module._trainer = SimpleNamespace(global_rank=0)

    with patch.object(module, "optimizers", return_value=SimpleNamespace(param_groups=[{"lr": 0.01}])):
        with patch.object(module, "log") as log:
            loss = module.training_step(make_batch(3))

    assert loss.requires_grad
    assert [call.args[0] for call in log.call_args_list] == ["learning_rate_epoch", "train_loss_epoch"]
    for call in log.call_args_list:
        assert call.kwargs["on_step"] is False
        assert call.kwargs["on_epoch"] is True
        assert call.kwargs["sync_dist"] is True
        assert call.kwargs["batch_size"] == 3


def test_epoch_mode_uses_named_batch_feature_for_shared_tensors():
    module = LightningModule(
        LossModel(1.0),
        epoch_only_training_metrics=True,
        training_batch_size_feature_name="item_id",
    )
    module._trainer = SimpleNamespace(global_rank=0)
    batch = make_batch(3)
    batch["feature_tensors"]["shared_negatives"] = torch.arange(50)

    with patch.object(module, "optimizers", return_value=SimpleNamespace(param_groups=[{"lr": 0.01}])):
        with patch.object(module, "log") as log:
            module.training_step(batch)

    assert all(call.kwargs["batch_size"] == 3 for call in log.call_args_list)


def test_epoch_mode_reports_metrics_with_lightning_trainer():
    module = LightningModule(ConstantLossModel(2.0), epoch_only_training_metrics=True)
    dataloader = DataLoader(
        [{"feature_tensors": {"item_id": torch.tensor([item_id])}} for item_id in range(5)],
        batch_size=2,
    )
    trainer = lightning.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(module, train_dataloaders=dataloader)

    torch.testing.assert_close(trainer.callback_metrics["train_loss_epoch"], torch.tensor(4.0))
    assert "learning_rate_epoch" in trainer.callback_metrics
