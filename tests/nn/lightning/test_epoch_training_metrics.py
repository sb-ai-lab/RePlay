import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lightning
import pytest
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


class BatchMeanLossModel(LossModel):
    def forward(self, feature_tensors):
        return {"loss": self.scale * 0 + feature_tensors["value"].float().mean()}


def make_batch(batch_size):
    return {
        "feature_tensors": {
            "item_id": torch.zeros(batch_size, 4, dtype=torch.long),
            "timestamp": torch.ones(batch_size, 4, dtype=torch.long),
        },
        "negative_labels": torch.arange(40_000),
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _distributed_epoch_metrics_worker(rank: int, port: int, output_dir: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    module = LightningModule(LossModel(1.0), epoch_only_training_metrics=True)
    module._trainer = SimpleNamespace(global_rank=rank)
    module._train_loss_sum.fill_(10 + 2 * rank)
    module._learning_rate_sum.fill_(0.2 + 0.4 * rank)
    module._training_row_count.fill_(2 + rank)
    reduction_count = 0
    original_all_reduce = torch.distributed.all_reduce

    def counted_all_reduce(*args, **kwargs):
        nonlocal reduction_count
        reduction_count += 1
        return original_all_reduce(*args, **kwargs)

    with patch.object(torch.distributed, "all_reduce", side_effect=counted_all_reduce):
        with patch.object(module, "log") as log:
            module.on_train_epoch_end()

    logged = {call.args[0]: float(call.args[1]) for call in log.call_args_list}
    torch.save((reduction_count, logged), Path(output_dir) / f"rank-{rank}.pt")
    torch.distributed.destroy_process_group()


def test_default_training_step_keeps_existing_logging():
    module = LightningModule(LossModel(2.0))
    module._trainer = SimpleNamespace(global_rank=0)

    with patch.object(module, "optimizers", return_value=SimpleNamespace(param_groups=[{"lr": 0.01}])):
        with patch.object(module, "log") as log:
            loss = module.training_step(make_batch(2))

    assert loss.requires_grad
    assert [call.args[0] for call in log.call_args_list] == ["learning_rate", "train_loss"]
    assert all(call.kwargs["sync_dist"] is True for call in log.call_args_list)


def test_epoch_mode_uses_rank_zero_step_metrics_without_synchronization():
    module = LightningModule(ConstantLossModel(2.0), epoch_only_training_metrics=True)
    module._trainer = SimpleNamespace(global_rank=0)

    with patch.object(module, "optimizers", return_value=SimpleNamespace(param_groups=[{"lr": 0.01}])):
        with patch.object(module, "log") as log:
            loss = module.training_step(make_batch(3))

    assert loss.requires_grad
    assert [call.args[0] for call in log.call_args_list] == ["learning_rate_step", "train_loss_step"]
    for call in log.call_args_list:
        assert call.kwargs["on_step"] is True
        assert call.kwargs["on_epoch"] is False
        assert call.kwargs["sync_dist"] is False
        assert call.kwargs["rank_zero_only"] is True
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
    module = LightningModule(BatchMeanLossModel(2.0), epoch_only_training_metrics=True)
    dataloader = DataLoader(
        [{"feature_tensors": {"value": torch.tensor(float(item_id))}} for item_id in range(5)],
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

    torch.testing.assert_close(trainer.callback_metrics["train_loss_epoch"], torch.tensor(2.0, dtype=torch.float64))
    assert "learning_rate_epoch" in trainer.callback_metrics


def test_epoch_mode_infers_batch_size_from_top_level_tensors():
    module = LightningModule(LossModel(1.0), epoch_only_training_metrics=True)

    assert module._infer_training_batch_size({"item_id": torch.zeros(3, 4, dtype=torch.long)}) == 3


def test_epoch_mode_reduces_metrics_once_in_two_process_ddp():
    with tempfile.TemporaryDirectory() as output_dir:
        torch.multiprocessing.spawn(
            _distributed_epoch_metrics_worker,
            args=(_free_port(), output_dir),
            nprocs=2,
            join=True,
        )
        results = {rank: torch.load(Path(output_dir) / f"rank-{rank}.pt", weights_only=True) for rank in range(2)}

    assert results[0][0] == results[1][0] == 1
    assert results[0][1]["train_loss_epoch"] == pytest.approx(4.4)
    assert results[0][1]["learning_rate_epoch"] == pytest.approx(0.16)
    assert results[1][1] == {}
