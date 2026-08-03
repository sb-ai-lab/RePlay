import pickle

import pytest
import torch

from replay.nn.transform import SparseUniformNegativeSamplingTransform, UniformNegativeSamplingTransform


def test_sparse_distribution_samples_only_positive_weight_items():
    transform = SparseUniformNegativeSamplingTransform(
        cardinality=10,
        num_negative_samples=3,
        sample_distribution=torch.tensor([0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0]),
        generator=torch.Generator().manual_seed(19),
    )

    output = transform({})

    assert set(output["negative_labels"].tolist()).issubset({1, 3, 6, 8})
    assert torch.unique(output["negative_labels"]).numel() == 3
    torch.testing.assert_close(transform.candidate_ids, torch.tensor([1, 3, 6, 8]))
    assert transform.sample_distribution.numel() == 4


def test_sparse_transform_preserves_generator_stream():
    compact_weights = torch.tensor([1.0, 4.0, 2.0, 3.0])
    expected_generator = torch.Generator().manual_seed(777)
    transform = SparseUniformNegativeSamplingTransform(
        cardinality=10,
        num_negative_samples=3,
        sample_distribution=torch.tensor([0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0]),
        generator=torch.Generator().manual_seed(777),
    )
    candidate_ids = torch.tensor([1, 3, 6, 8])

    for _ in range(2):
        expected = torch.take(
            candidate_ids,
            torch.multinomial(compact_weights, 3, replacement=False, generator=expected_generator),
        )
        torch.testing.assert_close(transform({})["negative_labels"], expected)


def test_sparse_transform_uses_dense_distribution_without_item_mapping():
    weights = torch.arange(1, 21, dtype=torch.float32)
    expected = UniformNegativeSamplingTransform(
        cardinality=20,
        num_negative_samples=7,
        sample_distribution=weights.clone(),
        generator=torch.Generator().manual_seed(777),
    )
    actual = SparseUniformNegativeSamplingTransform(
        cardinality=20,
        num_negative_samples=7,
        sample_distribution=weights,
        generator=torch.Generator().manual_seed(777),
    )

    assert actual.candidate_ids is None
    assert actual.sample_distribution.data_ptr() == weights.data_ptr()
    for _ in range(2):
        torch.testing.assert_close(actual({})["negative_labels"], expected({})["negative_labels"])


def test_sparse_transform_restores_checkpoint_and_generator_stream():
    weights = torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0])
    transform = SparseUniformNegativeSamplingTransform(
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


def test_sparse_transform_detaches_compacted_weights():
    weights = torch.tensor([0.0, 1.0, 0.0, 2.0], requires_grad=True)
    transform = SparseUniformNegativeSamplingTransform(
        cardinality=4,
        num_negative_samples=2,
        sample_distribution=weights,
    )

    assert not transform.sample_distribution.requires_grad
    assert transform.sample_distribution.grad_fn is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cardinality": 0, "num_negative_samples": 1}, "cardinality must be positive"),
        ({"cardinality": 4, "num_negative_samples": 0}, "num_negative_samples must be positive"),
        (
            {"cardinality": 4, "num_negative_samples": 2, "sample_distribution": torch.ones(3)},
            "incorrect size",
        ),
        ({"cardinality": 4, "num_negative_samples": 4}, "less than catalog cardinality"),
        (
            {
                "cardinality": 4,
                "num_negative_samples": 2,
                "sample_distribution": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "positive-weight candidates",
        ),
    ],
)
def test_sparse_transform_rejects_invalid_sampling_contract(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SparseUniformNegativeSamplingTransform(**kwargs)


@pytest.mark.parametrize(
    ("sample_distribution", "error", "message"),
    [
        (torch.ones((1, 4)), ValueError, "one-dimensional"),
        (torch.ones(4, dtype=torch.long), TypeError, "floating-point"),
        (torch.tensor([1.0, -1.0, 2.0, 3.0]), ValueError, "finite non-negative"),
        (torch.tensor([1.0, float("nan"), 2.0, 3.0]), ValueError, "finite non-negative"),
        (torch.tensor([1.0, float("inf"), 2.0, 3.0]), ValueError, "finite non-negative"),
    ],
)
def test_sparse_transform_rejects_invalid_distribution(sample_distribution, error, message):
    with pytest.raises(error, match=message):
        SparseUniformNegativeSamplingTransform(
            cardinality=4,
            num_negative_samples=2,
            sample_distribution=sample_distribution,
        )
