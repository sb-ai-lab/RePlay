from types import SimpleNamespace

import torch

from replay.nn.lightning import CatalogCacheLightningModule


def test_accumulation_window_end_includes_short_final_window():
    module = CatalogCacheLightningModule(torch.nn.Linear(2, 2))
    module._trainer = SimpleNamespace(accumulate_grad_batches=3, num_training_batches=5)

    assert not module._is_accumulation_window_end(0)
    assert not module._is_accumulation_window_end(1)
    assert module._is_accumulation_window_end(2)
    assert not module._is_accumulation_window_end(3)
    assert module._is_accumulation_window_end(4)
