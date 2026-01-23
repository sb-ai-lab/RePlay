import pytest

from replay.nn.mask import DefaultAttentionMask


@pytest.mark.parametrize("num_heads", [1, 2])
def test_default_mask_forward(simple_batch, num_heads):
    mask_builder = DefaultAttentionMask("item_id", num_heads=num_heads)

    mask = mask_builder(simple_batch["feature_tensors"], simple_batch["padding_mask"])

    batch_size, seq_len = simple_batch["feature_tensors"]["item_id"].shape
    assert mask.shape == (batch_size * num_heads, seq_len, seq_len)
