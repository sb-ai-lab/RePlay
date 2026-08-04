from typing import Any

import torch

from replay.nn.lightning.module import LightningModule


class CatalogCacheLightningModule(LightningModule):
    """Lightning wrapper for a loss that caches the item catalog across accumulation."""

    def _is_accumulation_window_end(self, batch_idx: int) -> bool:
        accumulation_steps = self.trainer.accumulate_grad_batches
        is_last_batch = batch_idx + 1 == self.trainer.num_training_batches
        return (batch_idx + 1) % accumulation_steps == 0 or is_last_batch

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        accumulation_steps = self.trainer.accumulate_grad_batches
        self.model.loss.prepare_accumulation_microbatch(
            is_window_end=self._is_accumulation_window_end(batch_idx),
            trainer_accumulation_steps=accumulation_steps,
            global_step=self.global_step,
        )
        return super().training_step(batch)

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        if "retain_graph" in kwargs:
            msg = "CatalogCacheLightningModule controls retain_graph during accumulation."
            raise ValueError(msg)
        kwargs["retain_graph"] = self.model.loss.should_retain_graph_for_current_backward()
        super().backward(loss, *args, **kwargs)

    def on_after_backward(self) -> None:
        self.model.loss.assert_current_backward_completed()
        super().on_after_backward()

    def on_train_epoch_start(self) -> None:
        self.model.loss.assert_accumulation_cache_idle()
        super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        self.model.loss.assert_accumulation_cache_idle()
        super().on_train_epoch_end()

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        self.model.loss.assert_accumulation_cache_idle()
        super().on_before_optimizer_step(optimizer)

    def on_validation_epoch_start(self) -> None:
        self.model.loss.assert_accumulation_cache_idle()
        super().on_validation_epoch_start()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.model.loss.assert_accumulation_cache_idle()
        super().on_save_checkpoint(checkpoint)
