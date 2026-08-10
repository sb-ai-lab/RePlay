import pytest
import torch

from replay.nn.sequential import LengthBucketedQueryEncoder, SasRecTransformerLayer


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


def test_bucketed_encoder_rejects_invalid_attention_mask_dimensions():
    with pytest.raises(ValueError, match="sequence"):
        LengthBucketedQueryEncoder._slice_attention_mask(
            torch.zeros(3, 4),
            torch.tensor([0]),
            left_crop=0,
            batch_size=1,
            sequence_length=4,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"bucket_size": 0}, "bucket_size"),
        ({"bucket_size": 1, "prediction_shift": -1}, "prediction_shift"),
        ({"bucket_size": 1, "min_input_elements": -1}, "min_input_elements"),
    ],
)
def test_bucketed_encoder_validates_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LengthBucketedQueryEncoder(ToyEncoder(2), **kwargs)


def test_bucketed_encoder_resets_wrapped_encoder():
    encoder = ToyEncoder(2)
    reset_calls = []
    encoder.reset_parameters = lambda: reset_calls.append(None)

    LengthBucketedQueryEncoder(encoder, bucket_size=1).reset_parameters()

    assert reset_calls == [None]


def test_bucketed_encoder_slices_shared_features_and_attention_mask():
    features = {
        "sequence": torch.arange(12).reshape(3, 4),
        "shared": torch.arange(4),
        "scalar": torch.tensor(1),
    }
    row_indices = torch.tensor([2, 0])

    actual_features = LengthBucketedQueryEncoder._slice_features(
        features,
        row_indices,
        left_crop=2,
        batch_size=3,
        sequence_length=4,
    )
    actual_mask = LengthBucketedQueryEncoder._slice_attention_mask(
        torch.ones(4, 4),
        row_indices,
        left_crop=1,
        batch_size=3,
        sequence_length=4,
    )

    torch.testing.assert_close(actual_features["sequence"], torch.tensor([[10, 11], [2, 3]]))
    assert actual_features["shared"] is features["shared"]
    assert actual_features["scalar"] is features["scalar"]
    assert actual_mask.shape == (3, 3)


def test_bucketed_encoder_rejects_invalid_input_shapes():
    encoder = LengthBucketedQueryEncoder(ToyEncoder(1), bucket_size=1, min_input_elements=0)
    attention_mask = torch.ones(3, 3)

    with pytest.raises(ValueError, match="input_embeddings"):
        encoder({}, torch.ones(2, 3), torch.ones(2, 3, dtype=torch.bool), attention_mask)
    with pytest.raises(ValueError, match="padding_mask"):
        encoder({}, torch.ones(2, 3, 1), torch.ones(2, 2, dtype=torch.bool), attention_mask)
    with pytest.raises(ValueError, match="non-empty"):
        encoder({}, torch.ones(0, 3, 1), torch.ones(0, 3, dtype=torch.bool), attention_mask)


def test_bucketed_encoder_rejects_changed_encoder_shape():
    class WrongShapeEncoder(ToyEncoder):
        def forward(self, feature_tensors, input_embeddings, padding_mask, attention_mask):
            return input_embeddings[:, :-1]

    features, embeddings, padding_mask, attention_mask = make_batch([2], 3, 2)
    encoder = LengthBucketedQueryEncoder(WrongShapeEncoder(2), bucket_size=1, min_input_elements=0)

    with pytest.raises(ValueError, match="preserve"):
        encoder(features, embeddings, padding_mask, attention_mask)


def test_bucketed_sasrec_matches_full_batch_with_flattened_multihead_mask():
    torch.manual_seed(8)
    features, baseline_embeddings, padding_mask, attention_mask = make_batch([2, 6, 3, 5], 7, 4)
    baseline_embeddings.requires_grad_(True)
    bucketed_embeddings = baseline_embeddings.detach().clone().requires_grad_(True)
    baseline_encoder = SasRecTransformerLayer(embedding_dim=4, num_heads=2, num_blocks=1, dropout=0.0)
    bucketed_encoder = SasRecTransformerLayer(embedding_dim=4, num_heads=2, num_blocks=1, dropout=0.0)
    bucketed_encoder.load_state_dict(baseline_encoder.state_dict())
    wrapper = LengthBucketedQueryEncoder(bucketed_encoder, bucket_size=2, min_input_elements=0)
    multihead_mask = attention_mask[:, 0].repeat_interleave(2, dim=0)

    baseline_output = baseline_encoder(features, baseline_embeddings, padding_mask, multihead_mask)
    bucketed_output = wrapper(features, bucketed_embeddings, padding_mask, multihead_mask)
    active = padding_mask.unsqueeze(-1).expand_as(baseline_output)
    torch.testing.assert_close(bucketed_output[active], baseline_output[active], rtol=1e-5, atol=1e-6)

    baseline_output[active].square().sum().backward()
    bucketed_output[active].square().sum().backward()
    torch.testing.assert_close(bucketed_embeddings.grad, baseline_embeddings.grad, rtol=1e-5, atol=1e-6)
