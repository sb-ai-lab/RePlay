import pytest
import torch
from replay.nn.sequential.sasrec import DiffTransformerLayer
from replay.nn.mask import DefaultAttentionMask



@pytest.mark.torch
def test_diff_attention_forward(simple_batch):
    mask_builder = DefaultAttentionMask("item_id", 2)
    diff_attn_block = DiffTransformerLayer(64, 2, 2)

    mask = mask_builder(simple_batch["feature_tensors"], simple_batch["padding_mask"])

    attn_hidden = diff_attn_block(
        feature_tensors=simple_batch["feature_tensors"],
        input_embeddings=torch.rand(4, 5, 64),
        padding_mask=None,
        # padding_mask=simple_batch["padding_mask"],
        attention_mask=mask,
    )
    assert attn_hidden.shape == (4, 5, 64)

    # mask_reshaped = mask.view(4, 2, 5, 5)

    # attn_hidden_with_mask_reshaped = diff_attn_block(
    #     feature_tensors=simple_batch["feature_tensors"],
    #     input_embeddings=torch.rand(4, 5, 64),
    #     padding_mask=None,
    #     attention_mask=mask_reshaped,
    # )
    # assert attn_hidden_with_mask_reshaped.shape == (4, 5, 64)
