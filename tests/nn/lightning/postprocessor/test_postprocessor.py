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
    repeated_logits = postprocessor.on_prediction(batch=batch, logits=input_logits)

    items_seen_mask = items_seen_mask[:, candidates]

    assert input_logits.shape == processed_logits.shape
    assert (torch.isinf(processed_logits) == items_seen_mask).all()
    assert (torch.isinf(repeated_logits) == items_seen_mask).all()

    postprocessor.candidates = None
    assert postprocessor.candidates is None


def test_seen_items_filter_predict_not_contiguous_score(batch, items_seen_mask):
    input_logits = torch.rand(5, batch["query_id"].shape[0]).transpose(0, 1)

    postprocessor = SeenItemsFilter(item_count=5)
    processed_logits = postprocessor.on_prediction(batch=batch, logits=input_logits)

    assert input_logits.shape == processed_logits.shape
    assert (torch.isinf(processed_logits) == items_seen_mask).all()


def test_seen_items_filter_handles_empty_full_and_repeated_histories():
    postprocessor = SeenItemsFilter(item_count=4)
    processed_logits = postprocessor.on_validation(
        {
            "seen_ids": torch.tensor(
                [
                    [0, 1, 2, 3],
                    [-1, -1, -1, -1],
                    [1, 1, -1, -1],
                    [1, 1, -1, -1],
                ]
            )
        },
        torch.rand(4, 4),
    )

    assert torch.isinf(processed_logits[0]).all()
    assert not torch.isinf(processed_logits[1]).any()
    assert torch.isinf(processed_logits[2:, 1]).all()
    assert not torch.isinf(processed_logits[2:, [0, 2, 3]]).any()


def test_seen_items_filter_moves_candidate_lookup_to_logits_device():
    postprocessor = SeenItemsFilter(item_count=4)
    postprocessor.candidates = torch.tensor([1, 3])

    lookup = postprocessor._get_candidate_lookup(torch.device("meta"))

    assert lookup.device.type == "meta"


@pytest.mark.parametrize("item_count", [0, -1])
def test_seen_items_filter_rejects_non_positive_item_count(item_count):
    with pytest.raises(ValueError, match="item_count must be positive"):
        SeenItemsFilter(item_count=item_count)


@pytest.mark.parametrize(
    "candidates",
    [
        torch.ones(1, 1, 1, dtype=torch.long),
        torch.tensor([1.0, 2.0]),
        torch.tensor([], dtype=torch.long),
        torch.tensor([1, 1]),
        torch.tensor([-1, 1]),
        torch.tensor([1, 5]),
    ],
)
def test_seen_items_filter_rejects_invalid_candidates(candidates):
    postprocessor = SeenItemsFilter(item_count=5)

    with pytest.raises((TypeError, ValueError)):
        postprocessor.candidates = candidates


@pytest.mark.parametrize(
    ("batch", "logits"),
    [
        ({"seen_ids": torch.tensor([[1]])}, torch.zeros(1)),
        ({"seen_ids": torch.tensor([[1]])}, torch.zeros(1, 5, dtype=torch.long)),
        ({"seen_ids": [[1]]}, torch.zeros(1, 5)),
        ({"seen_ids": torch.tensor([[1.0]])}, torch.zeros(1, 5)),
        ({"seen_ids": torch.tensor([1])}, torch.zeros(1, 5)),
        ({"seen_ids": torch.tensor([[1], [2]])}, torch.zeros(1, 5)),
        ({"seen_ids": torch.tensor([[1]])}, torch.zeros(1, 4)),
    ],
)
def test_seen_items_filter_rejects_invalid_inputs(batch, logits):
    postprocessor = SeenItemsFilter(item_count=5)

    with pytest.raises((TypeError, ValueError)):
        postprocessor.on_validation(batch, logits)


def test_seen_items_filter_rejects_row_wise_candidate_batch_mismatch():
    postprocessor = SeenItemsFilter(item_count=5)
    postprocessor.candidates = torch.tensor([[0, 1], [2, 3]])

    with pytest.raises(ValueError, match="same batch size"):
        postprocessor.on_validation({"seen_ids": torch.tensor([[1]])}, torch.zeros(1, 2))
