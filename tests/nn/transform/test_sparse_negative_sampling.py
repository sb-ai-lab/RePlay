import pickle

import pytest
import torch

from replay.nn.transform import UniformNegativeSamplingTransform


def test_sparse_distribution_samples_only_positive_weight_items():
    transform = UniformNegativeSamplingTransform(
        cardinality=10,
        num_negative_samples=3,
        sample_distribution=torch.tensor([0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0]),
        generator=torch.Generator().manual_seed(19),
    )

    output = transform({})

    assert set(output["negative_labels"].tolist()).issubset({1, 3, 6, 8})
    assert torch.unique(output["negative_labels"]).numel() == 3
    torch.testing.assert_close(transform._candidate_ids, torch.tensor([1, 3, 6, 8]))
    assert transform.sample_distribution.numel() == 4


def test_sparse_distribution_preserves_generator_stream():
    compact_weights = torch.tensor([1.0, 4.0, 2.0, 3.0])
    expected_generator = torch.Generator().manual_seed(777)
    transform = UniformNegativeSamplingTransform(
        cardinality=10,
        num_negative_samples=3,
        sample_distribution=torch.tensor([0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0]),
        generator=torch.Generator().manual_seed(777),
    )
    candidate_ids = torch.tensor([1, 3, 6, 8])

    for _ in range(2):
        compact_ids = torch.multinomial(compact_weights, 3, replacement=False, generator=expected_generator)
        expected = candidate_ids[compact_ids]
        torch.testing.assert_close(transform({})["negative_labels"], expected)


def test_dense_distribution_uses_original_sampling_path():
    weights = torch.arange(1, 21, dtype=torch.float32)
    expected_generator = torch.Generator().manual_seed(777)
    transform = UniformNegativeSamplingTransform(
        cardinality=20,
        num_negative_samples=7,
        sample_distribution=weights,
        generator=torch.Generator().manual_seed(777),
    )

    assert transform._candidate_ids is None
    assert transform.sample_distribution.data_ptr() == weights.data_ptr()
    for _ in range(2):
        expected = torch.multinomial(weights, 7, replacement=False, generator=expected_generator)
        torch.testing.assert_close(transform({})["negative_labels"], expected)


def test_sparse_distribution_restores_checkpoint_and_generator_stream():
    weights = torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0])
    transform = UniformNegativeSamplingTransform(
        cardinality=5,
        num_negative_samples=2,
        sample_distribution=weights,
        generator=torch.Generator().manual_seed(7),
    )
    transform({})

    restored = pickle.loads(pickle.dumps(transform))
    restored.load_state_dict(transform.state_dict(), strict=True)

    torch.testing.assert_close(restored({})["negative_labels"], transform({})["negative_labels"])
    assert torch.equal(restored.generator.get_state(), transform.generator.get_state())


def test_sparse_distribution_detaches_compacted_weights():
    weights = torch.tensor([0.0, 1.0, 0.0, 2.0], requires_grad=True)
    transform = UniformNegativeSamplingTransform(
        cardinality=4,
        num_negative_samples=2,
        sample_distribution=weights,
    )

    assert not transform.sample_distribution.requires_grad
    assert transform.sample_distribution.grad_fn is None


def test_sparse_distribution_rejects_insufficient_support():
    with pytest.raises(ValueError, match="positive-weight candidates"):
        UniformNegativeSamplingTransform(
            cardinality=4,
            num_negative_samples=2,
            sample_distribution=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )


@pytest.mark.parametrize(
    ("sample_distribution", "error", "message"),
    [
        (torch.ones(4, dtype=torch.long), TypeError, "floating-point"),
        (torch.tensor([1.0, -1.0, 2.0, 3.0]), ValueError, "finite non-negative"),
        (
            torch.tensor([1.0, float("nan"), 2.0, 3.0]),
            ValueError,
            "finite non-negative",
        ),
        (
            torch.tensor([1.0, float("inf"), 2.0, 3.0]),
            ValueError,
            "finite non-negative",
        ),
    ],
)
def test_sparse_distribution_rejects_invalid_weights(sample_distribution, error, message):
    with pytest.raises(error, match=message):
        UniformNegativeSamplingTransform(
            cardinality=4,
            num_negative_samples=2,
            sample_distribution=sample_distribution,
        )


def test_multidimensional_distribution_keeps_original_behavior():
    weights = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    expected_generator = torch.Generator().manual_seed(31)
    transform = UniformNegativeSamplingTransform(
        cardinality=4,
        num_negative_samples=2,
        sample_distribution=weights,
        generator=torch.Generator().manual_seed(31),
    )

    expected = torch.multinomial(weights, 2, replacement=False, generator=expected_generator)

    assert transform._candidate_ids is None
    assert transform.sample_distribution.data_ptr() == weights.data_ptr()
    torch.testing.assert_close(transform({})["negative_labels"], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sparse_distribution_moves_mapping_to_cuda():
    transform = UniformNegativeSamplingTransform(
        cardinality=5,
        num_negative_samples=2,
        sample_distribution=torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0]),
    ).cuda()

    output = transform({})["negative_labels"]

    assert transform.sample_distribution.is_cuda
    assert transform._candidate_ids.is_cuda
    assert output.is_cuda
    assert set(output.cpu().tolist()).issubset({1, 3, 4})
