import pytest
import torch

from replay.nn.transform import GroupedUniformNegativeSamplingTransform


def _make_transform(
    generator: torch.Generator,
) -> GroupedUniformNegativeSamplingTransform:
    return GroupedUniformNegativeSamplingTransform(
        cardinality=100,
        num_negative_samples=10,
        group_size=2,
        generator=generator,
    )


def test_grouped_sampler_preserves_independent_random_pools():
    grouped = _make_transform(torch.Generator().manual_seed(7))
    reference = _make_transform(torch.Generator().manual_seed(7))

    actual = grouped({"positive_labels": torch.zeros(8, 1, 1, dtype=torch.long)})["negative_labels"]
    expected = torch.stack(
        [reference({"positive_labels": torch.zeros(2, 1, 1, dtype=torch.long)})["negative_labels"][0] for _ in range(4)]
    )

    torch.testing.assert_close(actual, expected)


def test_grouped_sampler_draws_only_active_groups():
    grouped = _make_transform(torch.Generator().manual_seed(11))
    reference = _make_transform(torch.Generator().manual_seed(11))

    actual = grouped({"positive_labels": torch.zeros(3, 1, 1, dtype=torch.long)})["negative_labels"]
    first_reference = reference({"positive_labels": torch.zeros(4, 1, 1, dtype=torch.long)})["negative_labels"]
    next_actual = grouped({"positive_labels": torch.zeros(2, 1, 1, dtype=torch.long)})["negative_labels"][0]
    next_reference = reference({"positive_labels": torch.zeros(2, 1, 1, dtype=torch.long)})["negative_labels"][0]

    torch.testing.assert_close(actual[:2], first_reference[:2])
    assert actual.size(0) == 2
    torch.testing.assert_close(next_actual, next_reference)


def test_grouped_sampler_adds_explicit_group_size_to_batch():
    transform = _make_transform(torch.Generator().manual_seed(0))
    existing_feature = torch.ones(3, 1)

    output = transform(
        {
            "positive_labels": torch.zeros(3, 1, 1, dtype=torch.long),
            "feature_tensors": {"existing": existing_feature},
        }
    )

    assert output["feature_tensors"]["existing"] is existing_feature
    assert output["negative_group_size"] == transform.group_size
    assert set(output["feature_tensors"]) == {"existing"}


def test_grouped_sampler_restores_sparse_distribution_item_ids():
    transform = GroupedUniformNegativeSamplingTransform(
        cardinality=8,
        num_negative_samples=2,
        group_size=2,
        sample_distribution=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
        generator=torch.Generator().manual_seed(17),
    )

    pools = transform({"positive_labels": torch.zeros(4, 1, 1, dtype=torch.long)})["negative_labels"]

    assert set(pools.flatten().tolist()).issubset({4, 5, 6, 7})


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"group_size": 0}, "group_size"),
        (
            {
                "group_size": 2,
                "sample_distribution": torch.ones(2, 10),
            },
            "one-dimensional",
        ),
    ],
)
def test_grouped_sampler_validates_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GroupedUniformNegativeSamplingTransform(cardinality=10, num_negative_samples=2, **kwargs)


@pytest.mark.parametrize(
    "batch, message",
    [
        ({}, "does not contain"),
        ({"positive_labels": torch.tensor(1)}, "non-empty batch"),
        (
            {"positive_labels": torch.empty(0, 1, 1, dtype=torch.long)},
            "non-empty batch",
        ),
    ],
)
def test_grouped_sampler_validates_batch(batch, message):
    with pytest.raises(ValueError, match=message):
        _make_transform(torch.Generator().manual_seed(0))(batch)
