import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

    from replay.models.nn.sequential.sasrec import SasRecModel


@pytest.mark.torch
@pytest.mark.parametrize(
    "ti_modification",
    [
        (False),
        (True),
    ],
)
def test_sasrec_forward(tensor_schema, simple_masks, ti_modification):
    model = SasRecModel(
        tensor_schema.subset(["item_id", "timestamp"]), hidden_size=64, max_len=5, ti_modification=ti_modification
    )
    item_sequences, padding_mask, _, timestamp_sequences = simple_masks
    inputs = {"item_id": item_sequences, "timestamp": timestamp_sequences}

    assert model(inputs, padding_mask).size() == (4, 5, 4)


@pytest.mark.torch
def test_sasrec_predictions(tensor_schema, simple_masks):
    model = SasRecModel(tensor_schema.subset(["item_id"]), hidden_size=64, max_len=5)
    item_sequences, padding_mask, _, _ = simple_masks
    inputs = {
        "item_id": item_sequences,
    }

    predictions_by_one = model.predict(inputs, padding_mask, torch.tensor([0, 1, 2, 3]))
    predictions_all = model.predict(inputs, padding_mask)

    assert predictions_all.size() == predictions_by_one.size()


@pytest.mark.torch
def test_item_embedder_weights(tensor_schema):
    item_embedder = SasRecModel(
        tensor_schema.subset(["item_id", "timestamp"]), hidden_size=64, max_len=5, ti_modification=True
    ).item_embedder

    assert item_embedder.get_item_weights(torch.tensor([0, 1, 2, 3])).size() == (4, 64)


@pytest.mark.torch
def test_sasrec_forward_with_float_timematrix(tensor_schema, simple_masks):
    model = SasRecModel(tensor_schema.subset(["item_id", "timestamp"]), hidden_size=64, max_len=5, ti_modification=True)
    item_sequences, padding_mask, _, timestamp_sequences = simple_masks
    timestamp_sequences = timestamp_sequences.float()
    inputs = {"item_id": item_sequences, "timestamp": timestamp_sequences}

    assert model(inputs, padding_mask).size() == (4, 5, 4)
