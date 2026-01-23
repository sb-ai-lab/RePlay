import pytest
import torch

from replay.nn.lightning.postprocessor import SeenItemsFilter


@pytest.mark.parametrize("calling_method", ["on_validation", "on_prediction"])
def test_seen_items_filter_call(batch, items_seen_mask, calling_method):
    input_logits = torch.rand(batch["query_id"].shape[0], 5)

    postprocessor = SeenItemsFilter(item_count=5)

    processed_logits = getattr(postprocessor, calling_method)(batch=batch, logits=input_logits)

    assert input_logits.shape == processed_logits.shape
    assert (torch.isinf(processed_logits) == items_seen_mask).all()


@pytest.mark.parametrize(
    "candidates",
    [
        torch.LongTensor([0, 1]),
        torch.LongTensor([1, 3, 2, 4]),
    ],
)
def test_seen_items_filter_predict_with_candidates(batch, items_seen_mask, candidates):
    input_logits = torch.rand(batch["query_id"].shape[0], candidates.shape[0])

    postprocessor = SeenItemsFilter(item_count=5)
    postprocessor.candidates = candidates
    processed_logits = postprocessor.on_prediction(batch=batch, logits=input_logits)

    items_seen_mask = items_seen_mask[:, candidates]

    assert input_logits.shape == processed_logits.shape
    assert (torch.isinf(processed_logits) == items_seen_mask).all()


def test_seen_items_filter_predict_not_contiguous_score(batch, items_seen_mask):
    input_logits = torch.rand(5, batch["query_id"].shape[0]).transpose(0, 1)

    postprocessor = SeenItemsFilter(item_count=5)
    processed_logits = postprocessor.on_prediction(batch=batch, logits=input_logits)

    assert input_logits.shape == processed_logits.shape
    assert (torch.isinf(processed_logits) == items_seen_mask).all()
