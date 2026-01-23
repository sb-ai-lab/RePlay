import pytest
import torch

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def batch():
    query_id = torch.LongTensor([0, 1, 202, 3])
    seen_item_ids = torch.LongTensor(
        [
            [5, 5, 0, 1, 1],
            [1, 2, 4, 0, 3],
            [5, 5, 5, 5, 5],
            [0, 1, 2, 2, 2],
        ],
    )

    return {"query_id": query_id, "seen_ids": seen_item_ids}


@pytest.fixture(scope="module")
def items_seen_mask(batch):
    batch_size = batch["query_id"].shape[0]
    item_count_with_pad = batch["seen_ids"].max().item() + 1

    items_seen_mask = torch.full((batch_size, item_count_with_pad), False)
    is_seen = torch.full_like(batch["seen_ids"], True, dtype=torch.bool)
    items_seen_mask = items_seen_mask.scatter(dim=1, index=batch["seen_ids"], src=is_seen)

    return items_seen_mask[:, :-1]
