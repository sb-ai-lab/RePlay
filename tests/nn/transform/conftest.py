from typing import Optional

import pytest

pytest.importorskip("torch")
import torch


def make_random_tensor(
    batch_size: int, max_len: int, cardinality: int, pad_idx: int, fixed_padding_len: Optional[int] = None
):
    padded = torch.full((batch_size, max_len), pad_idx, dtype=torch.float32)
    for i in range(batch_size):
        if fixed_padding_len is not None:
            length = max_len - fixed_padding_len
        else:
            length = torch.randint(1, max_len + 1, (1,)).item()
        seq = torch.randint(1, cardinality, (length,), dtype=torch.long)
        padded[i, -length:] = seq
    mask = ~(padded == pad_idx)
    return padded, mask


def create_random_batch(
    batch_size: int,
    max_len: int,
    cardinality: int,
    num_classes: int,
    add_nested_keys: bool = False,
    fixed_padding_len: Optional[int] = None,
):
    batch = {}

    batch["item_id"], batch["item_id_mask"] = make_random_tensor(
        batch_size=batch_size,
        max_len=max_len,
        cardinality=cardinality,
        pad_idx=cardinality + 1,
        fixed_padding_len=fixed_padding_len,
    )
    batch["cat_feature"], batch["cat_feature_mask"] = make_random_tensor(
        batch_size=batch_size,
        max_len=max_len,
        cardinality=cardinality,
        pad_idx=cardinality + 1,
        fixed_padding_len=fixed_padding_len,
    )

    batch["user_id"] = torch.arange(batch_size)
    batch["user_id_mask"] = torch.full((batch_size,), 1, dtype=torch.bool)

    batch["negative_selector"] = torch.randint(0, num_classes, (batch_size,))

    if add_nested_keys:
        batch["group"] = {
            "feature_1": torch.rand((batch_size, max_len)),
            "feature_2": torch.rand((batch_size, max_len)),
        }

    return batch


@pytest.fixture(
    params=[
        {
            "batch_size": 5,
            "max_len": 10,
            "cardinality": 30,
            "num_classes": 2,
            "add_nested_keys": False,
            "fixed_padding_len": None,
        }
    ]
)
def random_batch(request):
    return create_random_batch(
        request.param.get("batch_size", 5),
        request.param.get("max_len", 10),
        request.param.get("cardinality", 30),
        request.param.get("num_classes", 2),
        request.param.get("add_nested_keys", False),
        request.param.get("fixed_padding_len", None),
    )


@pytest.fixture
def random_batch_with_nested_keys():
    return create_random_batch(batch_size=5, max_len=10, cardinality=30, num_classes=2, add_nested_keys=True)


@pytest.fixture
def random_batch_with_fixed_padding_len():
    return create_random_batch(
        batch_size=5, max_len=10, cardinality=30, num_classes=2, add_nested_keys=False, fixed_padding_len=3
    )


@pytest.fixture
def random_batch_with_full_seqs():
    return create_random_batch(
        batch_size=5, max_len=10, cardinality=30, num_classes=2, add_nested_keys=False, fixed_padding_len=0
    )
