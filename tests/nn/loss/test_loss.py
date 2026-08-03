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
from replay.nn.loss.base import weighted_mean


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


@pytest.mark.parametrize("loss", [CESampled(), BCESampled(), LogInCESampled()])
def test_sampled_loss_ignores_padded_negatives(loss):
    item_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    def get_logits(model_embeddings, item_ids):
        selected_item_embeddings = item_embeddings[item_ids]
        if item_ids.ndim == 1:
            return model_embeddings @ selected_item_embeddings.T
        return torch.einsum("bd,bnd->bn", model_embeddings, selected_item_embeddings)

    loss.logits_callback = get_logits
    batch = {
        "feature_tensors": {},
        "positive_labels": torch.tensor([[[0]]]),
        "padding_mask": torch.tensor([[True]]),
        "target_padding_mask": torch.tensor([[[True]]]),
    }

    embeddings_with_padding = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    with_padding = loss(
        model_embeddings=embeddings_with_padding,
        negative_labels=torch.tensor([1, -100, 2]),
        **batch,
    )
    with_padding.backward()

    embeddings_without_padding = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    without_padding = loss(
        model_embeddings=embeddings_without_padding,
        negative_labels=torch.tensor([1, 2]),
        **batch,
    )
    without_padding.backward()

    torch.testing.assert_close(with_padding, without_padding)
    torch.testing.assert_close(embeddings_with_padding.grad, embeddings_without_padding.grad)


def test_weighted_mean_clamps_zero_denominator():
    loss = torch.ones(2)
    sample_weight = torch.zeros_like(loss)

    torch.testing.assert_close(weighted_mean(loss, sample_weight), torch.tensor(0.0))


@pytest.mark.parametrize(
    ("weighted_loss_factory", "base_loss_factory", "sample_weight_mode"),
    [
        (
            lambda: CEWeighted(feature_name="sample_weight", ignore_index=3),
            lambda: CE(reduction="none", ignore_index=3),
            "flat",
        ),
        (
            lambda: CESampledWeighted(feature_name="sample_weight"),
            lambda: CESampled(reduction="none"),
            "masked",
        ),
        (
            lambda: LogOutCEWeighted(feature_name="sample_weight", cardinality=3),
            lambda: LogOutCE(cardinality=3, reduction="none"),
            "masked",
        ),
    ],
    ids=["CEWeighted", "CESampledWeighted", "LogOutCEWeighted"],
)
def test_weighted_losses_use_weighted_mean(
    weighted_loss_factory,
    base_loss_factory,
    sample_weight_mode,
    hidden_simple_weighted_batch,
    deterministic_logits_callback,
):
    loss = weighted_loss_factory()
    loss.logits_callback = deterministic_logits_callback
    base_loss = base_loss_factory()
    base_loss.logits_callback = deterministic_logits_callback
    sample_weight = hidden_simple_weighted_batch["feature_tensors"]["sample_weight"]

    if sample_weight_mode == "flat":
        sample_weight = sample_weight.view(-1)
    else:
        sample_weight = sample_weight[hidden_simple_weighted_batch["target_padding_mask"]]

    actual = loss(**hidden_simple_weighted_batch)
    expected = weighted_mean(base_loss(**hidden_simple_weighted_batch), sample_weight)

    torch.testing.assert_close(actual, expected)
