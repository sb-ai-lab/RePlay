import pytest
import torch

from replay.nn.head import EmbeddingTyingHead
from replay.nn.loss import BCE, CE, BCESampled, CESampled, LogInCE, LogInCESampled, LogOutCE


@pytest.mark.parametrize(
    "loss",
    [
        (CE(padding_idx=3)),
        (CESampled(padding_idx=3)),
        (BCE()),
        (BCESampled()),
        (LogInCE(vocab_size=3)),
        (LogInCESampled()),
        (LogOutCE(padding_idx=3, vocab_size=3)),
    ],
    ids=["CE loss", "CE sampled", "BCE", "BCE sampled", "LogInCE", "LogInCESampled", "LogOutCE"],
)
def test_setting_logits_callback_loss(loss, hidden_simple_batch):
    loss.logits_callback = None
    with pytest.raises(AttributeError):
        loss(**hidden_simple_batch)


@pytest.mark.parametrize(
    "loss",
    [
        (CE(padding_idx=3)),
        (CESampled(padding_idx=3)),
        (BCE()),
        (BCESampled()),
        (LogInCE(vocab_size=3)),
        (LogInCESampled()),
        (LogOutCE(padding_idx=3, vocab_size=3)),
    ],
    ids=["CE", "CE sampled", "BCE", "BCE sampled", "LogInCE", "LogInCESampled", "LogOutCE"],
)
@pytest.mark.parametrize(
    "batch_name",
    ["hidden_simple_batch", "hidden_simple_batch_multipositive"],
)
def test_loss_forward(loss, batch_name, request):

    def get_logits(dummy_hidden_out, dummy_item_emb=None):
        head = EmbeddingTyingHead()
        if dummy_item_emb is None:
            item_shape = (3,)
        else:
            item_shape = dummy_item_emb.shape

        item_emb = torch.rand(*item_shape, 32)
        return head(dummy_hidden_out, item_emb)

    loss.logits_callback = get_logits

    batch = request.getfixturevalue(batch_name)
    if isinstance(loss, CE) and batch_name == "hidden_simple_batch_multipositive":
        with pytest.raises(NotImplementedError):
            loss(**batch)
    else:
        loss(**batch)
