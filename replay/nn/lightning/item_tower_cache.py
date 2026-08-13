from typing import Any

import torch

from .module import LightningModule


class ItemTowerCacheLightningModule(LightningModule):
    """Lightning wrapper for a model that exposes its cache-owning tower via ``get_training_cache``."""

    _CACHE_METHODS = (
        "prepare_training_cache",
        "should_retain_training_cache_graph",
        "assert_training_cache_backward_completed",
        "assert_training_cache_idle",
    )

    def _training_cache(self) -> Any:
        cache_getter = getattr(self.model, "get_training_cache", None)
        if not callable(cache_getter):
            msg = "ItemTowerCacheLightningModule requires a model with get_training_cache()."
            raise TypeError(msg)
        training_cache = cache_getter()
        missing = tuple(name for name in self._CACHE_METHODS if not callable(getattr(training_cache, name, None)))
        if missing:
            msg = f"The training cache returned by the model is missing methods: {missing}."
            raise TypeError(msg)
        return training_cache

    def _is_accumulation_window_end(self, batch_idx: int) -> bool:
        accumulation_steps = self.trainer.accumulate_grad_batches
        is_last_batch = batch_idx + 1 == self.trainer.num_training_batches
        return (batch_idx + 1) % accumulation_steps == 0 or is_last_batch

    def training_step(self, batch: dict, batch_idx: int = 0) -> torch.Tensor:
        self._training_cache().prepare_training_cache(self._is_accumulation_window_end(batch_idx))
        return super().training_step(batch)

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        if "retain_graph" in kwargs:
            msg = "ItemTowerCacheLightningModule controls retain_graph during accumulation."
            raise ValueError(msg)
        retain_graph = self._training_cache().should_retain_training_cache_graph()
        kwargs["retain_graph"] = retain_graph
        super().backward(loss, *args, **kwargs)

    def on_after_backward(self) -> None:
        self._training_cache().assert_training_cache_backward_completed()
        super().on_after_backward()

    def on_train_epoch_start(self) -> None:
        self._training_cache().assert_training_cache_idle()
        super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        self._training_cache().assert_training_cache_idle()
        super().on_train_epoch_end()

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        self._training_cache().assert_training_cache_idle()
        super().on_before_optimizer_step(optimizer)

    def on_validation_epoch_start(self) -> None:
        self._training_cache().assert_training_cache_idle()
        super().on_validation_epoch_start()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self._training_cache().assert_training_cache_idle()
        super().on_save_checkpoint(checkpoint)

    def on_train_start(self) -> None:
        self._training_cache()
        if not self.automatic_optimization:
            msg = "ItemTowerCacheLightningModule requires automatic optimization."
            raise RuntimeError(msg)
        if self.trainer.strategy.handles_gradient_accumulation:
            msg = "The selected Lightning strategy handles gradient accumulation internally."
            raise RuntimeError(msg)
        super().on_train_start()
