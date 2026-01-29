import pytest
import torch

from replay.nn.head import EmbeddingTyingHead
from replay.nn.loss import (
    BCE,
    CE,
    BCESampled,
    CESampled,
    CESampledWeighted,
    CEWeighted,
    LogInCE,
    LogInCESampled,
    LogOutCE,
    LogOutCEWeighted,
)


@pytest.mark.parametrize(
    "loss",
    [
        (CE(ignore_index=3)),
        (CEWeighted(feature_name="sample_weights", ignore_index=3)),
        (CESampled(ignore_index=3)),
        (CESampledWeighted(feature_name="sample_weights", ignore_index=3)),
        (BCE()),
        (BCESampled()),
        (LogInCE(cardinality=3)),
        (LogInCESampled()),
        (LogOutCE(ignore_index=3, cardinality=3)),
        (LogOutCEWeighted(feature_name="sample_weights", ignore_index=3, cardinality=3)),
    ],
    ids=[
        "CE",
        "CE weighted",
        "CE sampled",
        "CE sampled & weighted",
        "BCE",
        "BCE sampled",
        "LogInCE",
        "LogInCE sampled",
        "LogOutCE",
        "LogOutCE weighted",
    ],
)
def test_setting_logits_callback_loss(loss, hidden_simple_batch):
    loss.logits_callback = None
    with pytest.raises(AttributeError):
        loss(**hidden_simple_batch)


@pytest.mark.parametrize(
    "loss",
    [
        (CE(ignore_index=3)),
        (CEWeighted(feature_name="sample_weight", ignore_index=3)),
        (CESampled(ignore_index=3)),
        (CESampled(ignore_index=3, negative_labels_ignore_index=0)),
        (CESampledWeighted(feature_name="sample_weight", ignore_index=3, negative_labels_ignore_index=0)),
        (BCE()),
        (BCESampled()),
        (LogInCE(cardinality=3)),
        (LogInCESampled()),
        (LogOutCE(ignore_index=3, cardinality=3)),
        (LogOutCEWeighted(feature_name="sample_weight", ignore_index=3, cardinality=3)),
    ],
    ids=[
        "CE",
        "CE weighted",
        "CE sampled",
        "CE sampled w/ negative ignore index",
        "CE sampled & weighted w/ negative ignore index",
        "BCE",
        "BCE sampled",
        "LogInCE",
        "LogInCE sampled",
        "LogOutCE",
        "LogOutCE weighted",
    ],
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
            _ = loss(**batch)
    else:
        _ = loss(**batch)


@pytest.mark.parametrize(
    "loss",
    [
        (CESampled(ignore_index=3)),
        (BCESampled()),
        (LogInCESampled()),
    ],
    ids=["CE sampled", "BCE sampled", "LogInCESampled"],
)
@pytest.mark.parametrize(
    "batch_name",
    ["hidden_simple_batch_multiclass_negatives", "hidden_simple_batch_multiclass_negatives_multipositive"],
)
def test_loss_forward_with_multiclass_negatives(loss, batch_name, request):
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

    loss(**batch)
