from types import SimpleNamespace
from unittest.mock import Mock

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from replay.nn.lightning import LightningModule
from replay.nn.lightning.callback import ComputeMetricsCallback
from replay.nn.lightning.postprocessor import SeenItemsFilter


class _RecordingMetricsBuilder:
    max_k = 2

    def __init__(self):
        self.predictions = []

    def add_prediction(self, predictions, ground_truth, train=None):
        self.predictions.append(predictions)


class _FixedCandidateModel(torch.nn.Module):
    def forward(self, query_id, candidates_to_score=None):
        assert candidates_to_score is not None
        logits = torch.tensor([0.8, 0.9, 0.1], device=query_id.device)
        return {"logits": logits.expand(query_id.shape[0], -1)}


def _callback_with_builder(candidates=None, postprocessors=None):
    callback = ComputeMetricsCallback(metrics=["recall"], ks=[2], item_count=8, postprocessors=postprocessors)
    callback._set_candidates(SimpleNamespace(candidates_to_score=candidates))
    builder = _RecordingMetricsBuilder()
    callback._metrics_builders = [builder]
    callback._dataloaders_size = [2]
    return callback, builder


def test_subset_topk_is_mapped_to_global_item_ids():
    callback, builder = _callback_with_builder(torch.tensor([4, 1, 3]))

    callback._batch_end(
        trainer=Mock(),
        pl_module=Mock(),
        outputs={"logits": torch.tensor([[0.8, 0.9, 0.1], [0.2, 0.3, 0.7]])},
        batch={"ground_truth": torch.tensor([[1], [3]])},
        batch_idx=0,
        dataloader_idx=0,
    )

    torch.testing.assert_close(builder.predictions[0], torch.tensor([[1, 4], [3, 1]]))


def test_batch_candidates_override_module_candidates():
    callback, builder = _callback_with_builder(torch.tensor([0, 1, 2]))

    callback._batch_end(
        trainer=Mock(),
        pl_module=Mock(),
        outputs={"logits": torch.tensor([[0.1, 0.9, 0.2]])},
        batch={
            "ground_truth": torch.tensor([[4]]),
            "candidates_to_score": torch.tensor([1, 4, 3]),
        },
        batch_idx=0,
        dataloader_idx=0,
    )

    torch.testing.assert_close(builder.predictions[0], torch.tensor([[4, 3]]))


def test_row_wise_candidates_are_mapped_and_seen_items_are_masked():
    postprocessor = SeenItemsFilter(item_count=8)
    callback, builder = _callback_with_builder(postprocessors=[postprocessor])

    callback._batch_end(
        trainer=Mock(),
        pl_module=Mock(),
        outputs={"logits": torch.tensor([[0.8, 0.9, 0.1], [0.2, 0.3, 0.7]])},
        batch={
            "ground_truth": torch.tensor([[4], [5]]),
            "seen_ids": torch.tensor([[1, 7], [2, -1]]),
            "candidates_to_score": torch.tensor([[4, 1, 3], [0, 5, 2]]),
        },
        batch_idx=0,
        dataloader_idx=0,
    )

    torch.testing.assert_close(builder.predictions[0], torch.tensor([[4, 3], [5, 0]]))


@pytest.mark.torch
def test_lightning_validation_maps_candidates_after_seen_item_filtering():
    callback = ComputeMetricsCallback(
        metrics=["recall"],
        ks=[1],
        item_count=8,
        postprocessors=[SeenItemsFilter(item_count=8)],
        verbose=False,
    )
    model = LightningModule(_FixedCandidateModel())
    model.candidates_to_score = torch.tensor([4, 1, 3])
    dataloader = DataLoader(
        [{"query_id": torch.tensor(0), "ground_truth": torch.tensor([4]), "seen_ids": torch.tensor([1])}],
        batch_size=1,
    )
    trainer = L.Trainer(
        callbacks=[callback],
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.validate(model, dataloaders=dataloader)

    assert callback.get_metrics()[0]["recall@1"] == 1.0


@pytest.mark.parametrize(
    "candidates",
    [
        torch.tensor([1, 1]),
        torch.tensor([-1, 2]),
        torch.tensor([1, 8]),
        torch.tensor([1.0, 2.0]),
    ],
)
def test_invalid_candidates_are_rejected(candidates):
    callback = ComputeMetricsCallback(metrics=["recall"], ks=[1], item_count=8)

    with pytest.raises((TypeError, ValueError)):
        callback._set_candidates(SimpleNamespace(candidates_to_score=candidates))
