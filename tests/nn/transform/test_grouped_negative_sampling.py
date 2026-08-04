import torch

from replay.nn.transform import GroupedUniformNegativeSamplingTransform


def _make_transform(generator: torch.Generator) -> GroupedUniformNegativeSamplingTransform:
    return GroupedUniformNegativeSamplingTransform(
        cardinality=100,
        num_negative_samples=10,
        group_size=2,
        groups_per_batch=4,
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


def test_grouped_sampler_does_not_advance_rng_for_missing_final_groups():
    grouped = _make_transform(torch.Generator().manual_seed(11))
    reference = _make_transform(torch.Generator().manual_seed(11))

    actual = grouped({"positive_labels": torch.zeros(3, 1, 1, dtype=torch.long)})["negative_labels"]
    first_reference = reference({"positive_labels": torch.zeros(4, 1, 1, dtype=torch.long)})["negative_labels"]
    next_actual = grouped({"positive_labels": torch.zeros(2, 1, 1, dtype=torch.long)})["negative_labels"][0]
    next_reference = reference({"positive_labels": torch.zeros(2, 1, 1, dtype=torch.long)})["negative_labels"][0]

    torch.testing.assert_close(actual[:2], first_reference[:2])
    torch.testing.assert_close(actual[2], actual[1])
    torch.testing.assert_close(actual[3], actual[1])
    torch.testing.assert_close(next_actual, next_reference)
