import pytest

pytest.importorskip("torch")
import torch


def make_random_tensor(
    batch_size: int,
    max_len: int,
    cardinality: int,
    pad_idx,
):
    padded = torch.full((batch_size, max_len), pad_idx, dtype=torch.float32)
    for i in range(batch_size):
        length = torch.randint(1, max_len + 1, (1,)).item()
        seq = torch.randint(1, cardinality, (length,), dtype=torch.long)
        padded[i, -length:] = seq
    mask = ~(padded == pad_idx)
    return padded, mask


def create_random_batch(batch_size: int, max_len: int, cardinality: int, num_classes: int):
    batch = {}

    batch["item_id"], batch["item_id_mask"] = make_random_tensor(
        batch_size=batch_size, max_len=max_len, cardinality=cardinality, pad_idx=cardinality + 1
    )
    batch["cat_feature"], batch["cat_feature_mask"] = make_random_tensor(
        batch_size=batch_size, max_len=max_len, cardinality=cardinality, pad_idx=cardinality + 1
    )
    batch["user_id"] = torch.arange(batch_size)
    batch["user_id_mask"] = torch.full((batch_size,), 1, dtype=torch.bool)

    batch["negative_selector"] = torch.randint(0, num_classes, (batch_size,))

    return batch


@pytest.fixture(params=[{"batch_size": 5, "max_len": 10, "cardinality": 30, "num_classes": 2}])
def random_batch(request):
    return create_random_batch(
        request.param["batch_size"],
        request.param["max_len"],
        request.param["cardinality"],
        request.param.get("num_classes", 2),
    )
