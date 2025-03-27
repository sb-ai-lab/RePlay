import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

    from replay.models.nn.sequential.sasrec_with_llm import SasRecLLMModel


@pytest.mark.torch
@pytest.mark.parametrize(
    "ti_modification",
    [
        (False),
        (True),
    ],
)
def test_sasrec_forward(tensor_schema, simple_masks, ti_modification):
    model = SasRecLLMModel(
        tensor_schema.subset(["item_id", "timestamp"]),
        profile_emb_dim=1024,
        hidden_size=64,
        max_len=5,
        ti_modification=ti_modification,
    )
    item_sequences, padding_mask, _, timestamp_sequences = simple_masks
    inputs = {"item_id": item_sequences, "timestamp": timestamp_sequences}
    output, hidden_states = model(inputs, padding_mask)
    assert output.size() == (4, 5, 4)


@pytest.mark.torch
def test_sasrec_predictions(tensor_schema, simple_masks):
    model = SasRecLLMModel(tensor_schema.subset(["item_id"]), profile_emb_dim=1024, hidden_size=64, max_len=5)
    item_sequences, padding_mask, _, _ = simple_masks
    inputs = {
        "item_id": item_sequences,
    }

    candidates_to_score = torch.range(0, tensor_schema["item_id"].cardinality - 1, 1, dtype=torch.long)
    predictions_all_candidates = model.predict(inputs, padding_mask, candidates_to_score)
    predictions_all = model.predict(inputs, padding_mask)
    assert predictions_all.size() == predictions_all_candidates.size()

    candidates_to_score = torch.tensor([0, 1])
    predictions_candidates = model.predict(inputs, padding_mask, candidates_to_score)
    assert predictions_candidates.size() == torch.Size([padding_mask.shape[0], candidates_to_score.shape[0]])


def test_predictions_equal_with_permuted_candidates(tensor_schema, simple_masks):
    model = SasRecLLMModel(tensor_schema.subset(["item_id"]), profile_emb_dim=1024, hidden_size=64, max_len=5)
    item_sequences, padding_mask, _, _ = simple_masks
    inputs = {
        "item_id": item_sequences,
    }
    sorted_candidates = torch.LongTensor([0, 1, 2, 3])
    permuted_candidates = torch.LongTensor([3, 0, 2, 1])
    _, ordering = torch.sort(permuted_candidates)
    model.eval()

    predictions_sorted_candidates = model.predict(inputs, padding_mask, sorted_candidates)
    predictions_permuted_candidates = model.predict(inputs, padding_mask, permuted_candidates)
    assert torch.equal(predictions_permuted_candidates[:, ordering], predictions_sorted_candidates)


@pytest.mark.torch
def test_item_embedder_weights(tensor_schema):
    item_embedder = SasRecLLMModel(
        tensor_schema.subset(["item_id", "timestamp"]),
        profile_emb_dim=1024,
        hidden_size=64,
        max_len=5,
        ti_modification=True,
    ).item_embedder

    assert item_embedder.get_item_weights(torch.tensor([0, 1, 2, 3])).size() == (4, 64)


@pytest.mark.torch
def test_sasrec_forward_with_float_timematrix(tensor_schema, simple_masks):
    model = SasRecLLMModel(
        tensor_schema.subset(["item_id", "timestamp"]),
        profile_emb_dim=1024,
        hidden_size=64,
        max_len=5,
        ti_modification=True,
    )
    item_sequences, padding_mask, _, timestamp_sequences = simple_masks
    timestamp_sequences = timestamp_sequences.float()
    inputs = {"item_id": item_sequences, "timestamp": timestamp_sequences}
    output, hidden_states = model(inputs, padding_mask)
    assert output.size() == (4, 5, 4)
