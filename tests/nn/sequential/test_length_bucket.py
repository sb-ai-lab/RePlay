import pytest
import torch

from replay.nn.sequential import LengthBucketedQueryEncoder


class ToyEncoder(torch.nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.input_shapes = []

    def forward(self, feature_tensors, input_embeddings, padding_mask, attention_mask):
        self.input_shapes.append(tuple(input_embeddings.shape))
        assert feature_tensors["timestamp"].shape == padding_mask.shape
        assert feature_tensors["request_id"].shape == padding_mask.shape[:1]
        weights = torch.softmax(attention_mask[:, 0], dim=-1)
        signal = feature_tensors["timestamp"].to(input_embeddings.dtype).unsqueeze(-1)
        return self.projection(torch.bmm(weights, input_embeddings)) + signal * 1e-4


def make_batch(lengths, sequence_length, embedding_dim):
    positions = torch.arange(sequence_length).unsqueeze(0)
    padding_mask = positions >= sequence_length - torch.tensor(lengths).unsqueeze(1)
    batch_size = len(lengths)
    embeddings = torch.randn(batch_size, sequence_length, embedding_dim)
    timestamps = torch.arange(sequence_length).expand(batch_size, -1) + 1_700_000_000
    features = {
        "timestamp": torch.where(padding_mask, timestamps, 0),
        "request_id": torch.arange(batch_size),
    }
    causal = torch.ones(sequence_length, sequence_length, dtype=torch.bool).tril()
    diagonal = torch.eye(sequence_length, dtype=torch.bool)
    allowed = causal[None] & (padding_mask[:, None, :] | diagonal[None])
    attention_mask = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, -torch.inf)[:, None]
    return features, embeddings, padding_mask, attention_mask


def test_bucketed_active_outputs_and_gradients_match_full_batch():
    torch.manual_seed(2)
    features, baseline_embeddings, padding_mask, attention_mask = make_batch([2, 6, 3, 5, 1, 4], 7, 4)
    baseline_embeddings.requires_grad_(True)
    bucketed_embeddings = baseline_embeddings.detach().clone().requires_grad_(True)
    baseline_encoder = ToyEncoder(4)
    bucketed_encoder = ToyEncoder(4)
    bucketed_encoder.load_state_dict(baseline_encoder.state_dict())
    wrapper = LengthBucketedQueryEncoder(bucketed_encoder, bucket_size=2, min_input_elements=0)

    baseline_output = baseline_encoder(features, baseline_embeddings, padding_mask, attention_mask)
    bucketed_output = wrapper(features, bucketed_embeddings, padding_mask, attention_mask)
    active = padding_mask.unsqueeze(-1).expand_as(baseline_output)
    torch.testing.assert_close(bucketed_output[active], baseline_output[active])

    baseline_output[active].square().sum().backward()
    bucketed_output[active].square().sum().backward()
    torch.testing.assert_close(bucketed_embeddings.grad, baseline_embeddings.grad)
    torch.testing.assert_close(
        bucketed_encoder.projection.weight.grad,
        baseline_encoder.projection.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    assert bucketed_encoder.input_shapes == [(2, 3, 4), (2, 5, 4), (2, 7, 4)]


def test_bucketed_encoder_keeps_shifted_target_boundary():
    torch.manual_seed(4)
    features, embeddings, padding_mask, attention_mask = make_batch([1, 3, 5], 6, 3)
    baseline_encoder = ToyEncoder(3)
    bucketed_encoder = ToyEncoder(3)
    bucketed_encoder.load_state_dict(baseline_encoder.state_dict())
    wrapper = LengthBucketedQueryEncoder(bucketed_encoder, bucket_size=1, prediction_shift=1, min_input_elements=0)
    target_mask = torch.cat((padding_mask[:, 1:], padding_mask[:, -1:]), dim=1)

    expected = baseline_encoder(features, embeddings, padding_mask, attention_mask)
    actual = wrapper(features, embeddings, padding_mask, attention_mask)

    active = target_mask.unsqueeze(-1).expand_as(expected)
    torch.testing.assert_close(actual[active], expected[active])
    assert bucketed_encoder.input_shapes == [(1, 2, 3), (1, 4, 3), (1, 6, 3)]


def test_bucketed_encoder_bypasses_small_and_evaluation_batches():
    features, embeddings, padding_mask, attention_mask = make_batch([1, 3], 4, 2)
    encoder = ToyEncoder(2)
    wrapper = LengthBucketedQueryEncoder(encoder, bucket_size=1, min_input_elements=100).train()

    wrapper(features, embeddings, padding_mask, attention_mask)
    wrapper.min_input_elements = 0
    wrapper.eval()(features, embeddings, padding_mask, attention_mask)

    assert encoder.input_shapes == [(2, 4, 2), (2, 4, 2)]


def test_bucketed_encoder_slices_flattened_multihead_mask_by_row():
    batch_size, num_heads, sequence_length = 3, 2, 5
    mask = torch.arange(batch_size * num_heads * sequence_length**2).reshape(
        batch_size * num_heads,
        sequence_length,
        sequence_length,
    )
    row_indices = torch.tensor([2, 0])

    actual = LengthBucketedQueryEncoder._slice_attention_mask(
        mask,
        row_indices,
        left_crop=2,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    expected = mask.reshape(batch_size, num_heads, sequence_length, sequence_length)[row_indices, :, 2:, 2:]

    torch.testing.assert_close(actual, expected.reshape(-1, 3, 3))


def test_bucketed_encoder_rejects_unsupported_attention_mask_layout():
    with pytest.raises(ValueError, match="batch-aligned"):
        LengthBucketedQueryEncoder._slice_attention_mask(
            torch.zeros(1, 2, 4, 4),
            torch.tensor([0]),
            left_crop=0,
            batch_size=3,
            sequence_length=4,
        )
