import pytest

pytest.importorskip("torch")
import torch


def make_random_tensor(
    batch_size: int,
    max_len: int,
    vocab_size: int,
    pad_idx,
):
    padded = torch.full((batch_size, max_len), pad_idx, dtype=torch.float32)
    for i in range(batch_size):
        length = torch.randint(1, max_len + 1, (1,)).item()
        seq = torch.randint(1, vocab_size, (length,), dtype=torch.long)
        padded[i, -length:] = seq
    mask = ~(padded == pad_idx)
    return padded, mask


def create_random_batch(batch_size, max_len, vocab_size):
    batch = {}

    batch["item_id"], batch["item_id_mask"] = make_random_tensor(
        batch_size=batch_size, max_len=max_len, vocab_size=vocab_size, pad_idx=vocab_size + 1
    )
    batch["cat_feature"], batch["cat_feature_mask"] = make_random_tensor(
        batch_size=batch_size, max_len=max_len, vocab_size=vocab_size, pad_idx=vocab_size + 1
    )
    batch["user_id"] = torch.arange(batch_size)
    batch["user_id_mask"] = torch.full((batch_size,), 1, dtype=torch.bool)
    return batch


@pytest.fixture(params=[{"batch_size": 5, "max_len": 10, "vocab_size": 30}])
def random_batch(request):
    return create_random_batch(request.param["batch_size"], request.param["max_len"], request.param["vocab_size"])
