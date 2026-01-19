import copy

import pytest
import torch

from replay.models.nn.sequential.sasrec import SasRecTrainingBatch
from replay.nn.transforms import (
    BatchingTransform,
    CopyTransform,
    GroupTransform,
    NextTokenTransform,
    RenameTransform,
    SequenceRollTransform,
    TokenMaskTransform,
    TrimTransform,
    UniformNegativeSamplingTransform,
    UnsqueezeTransform,
)


@pytest.mark.parametrize(
    "random_batch",
    [
        ({"batch_size": 5, "max_len": 10, "vocab_size": 30}),
        ({"batch_size": 64, "max_len": 200, "vocab_size": 3}),
        ({"batch_size": 1024, "max_len": 500, "vocab_size": 300}),
    ],
    indirect=True,
)
@pytest.mark.parametrize("shift", [1, 5])
def test_next_token_label_transform(random_batch, shift):
    label_field = "item_id"
    query_field = "user_id"
    transform = NextTokenTransform(label_field=label_field, query_features=query_field, shift=shift)
    transformed_batch = transform(random_batch)

    for feature in random_batch.keys():
        if feature.startswith(query_field):
            torch.testing.assert_close(transformed_batch[feature], random_batch[feature])
        else:
            torch.testing.assert_close(transformed_batch[feature], random_batch[feature][:, :-shift])

    torch.testing.assert_close(transformed_batch["positive_labels"], random_batch[label_field][:, shift:])
    torch.testing.assert_close(transformed_batch["positive_labels_mask"], random_batch[f"{label_field}_mask"][:, shift:])


@pytest.mark.parametrize(
    "shift, batch",
    [
        (100, {"item_id": torch.tensor([[0]]), "user_id": torch.tensor([0])}),
        (1, {"item_id": torch.tensor([0, 1]), "user_id": torch.tensor([0])}),
    ],
    ids=["Incorrect shift value", "Incorrect tensor shape"],
)
def test_next_token_label_raises_error(batch, shift):
    with pytest.raises(ValueError):
        transform = NextTokenTransform(label_field="item_id", shift=shift)
        transform(batch)


def test_batching_transform_raises_error(random_batch):
    transform = BatchingTransform(SasRecTrainingBatch)
    with pytest.raises(ValueError):
        transform(random_batch)


def test_rename_transform(random_batch):
    transform = RenameTransform({"user_id": "query_id", "item_id_mask": "padding_mask"})

    source_batch = copy.deepcopy(random_batch)
    transformed_batch = transform(random_batch)

    assert transformed_batch["query_id"].equal(source_batch["user_id"])
    assert transformed_batch["padding_mask"].equal(source_batch["item_id_mask"])

    assert {"query_id", "padding_mask"}.isdisjoint(set(source_batch.keys()))
    assert {"user_id", "item_id_mask"}.isdisjoint(set(transformed_batch.keys()))


@pytest.mark.parametrize(
    "random_batch, vocab_size, num_negative_samples",
    [
        ({"batch_size": 5, "max_len": 10, "vocab_size": 30}, 30, 29),
        ({"batch_size": 64, "max_len": 200, "vocab_size": 3}, 3, 1),
    ],
    indirect=["random_batch"],
)
@pytest.mark.parametrize("generate_sample_distribution", [True, False])
def test_negative_sampling_transform(random_batch, vocab_size, num_negative_samples, generate_sample_distribution):
    sample_distribution = None
    if generate_sample_distribution:
        sample_distribution = torch.rand(vocab_size)

    transform = UniformNegativeSamplingTransform(
        vocab_size=vocab_size, num_negative_samples=num_negative_samples, sample_distribution=sample_distribution
    )

    transformed_batch = transform(random_batch)

    assert "negative_labels" in transformed_batch.keys()
    assert transformed_batch["negative_labels"].size() == (num_negative_samples,)


@pytest.mark.parametrize(
    "vocab_size, num_negative_samples, sample_distribution, expected_exception",
    [(10, 1, torch.rand(100), pytest.raises(ValueError)), (10, 100, None, pytest.raises(AssertionError))],
    ids=["Incorrect sample distribution size", "Incorrect num_negative_samples"],
)
def test_negative_sampling_raises(vocab_size, num_negative_samples, sample_distribution, expected_exception):
    with expected_exception:
        UniformNegativeSamplingTransform(
            vocab_size=vocab_size, num_negative_samples=num_negative_samples, sample_distribution=sample_distribution
        )


def test_unsqueeze_transform(random_batch):
    feature_to_unsqueeze = "cat_feature"

    source_shape = random_batch[feature_to_unsqueeze].shape
    transform = UnsqueezeTransform(feature_to_unsqueeze, -1)
    transformed_batch = transform(random_batch)

    assert transformed_batch[feature_to_unsqueeze].shape == (*source_shape, 1)


def test_unsqueeze_transform_raises_error(random_batch):
    transform = UnsqueezeTransform("cat_feature", 3)
    with pytest.raises(ValueError):
        transform(random_batch)


@pytest.mark.parametrize(
    "roll",
    [1, 5, -1, -5],
)
@pytest.mark.parametrize(
    "feature_to_roll, padding_value",
    [("item_id_mask", 1), ("item_id", 51)],
)
def test_rolling_transform(random_batch, roll, feature_to_roll, padding_value):
    source_tensor = random_batch[feature_to_roll].clone()

    transform = SequenceRollTransform(feature_to_roll, roll=roll, padding_value=padding_value)
    output_tensor = transform(random_batch)[feature_to_roll]

    if roll > 0:
        assert torch.equal(source_tensor[:, :-roll], output_tensor[:, roll:])
        assert torch.eq(output_tensor[:, :roll], padding_value).all().item()
    else:
        assert torch.equal(source_tensor[:, -roll:], output_tensor[:, :roll])
        assert torch.eq(output_tensor[:, roll:], padding_value).all().item()


def test_group_transform(random_batch):
    transform = GroupTransform({"features": ["item_id", "cat_feature"]})
    transformed_batch = transform(random_batch)

    assert "features" in transformed_batch.keys()
    assert {"item_id", "cat_feature"} == set(transformed_batch["features"].keys())
    assert {"item_id", "cat_feature"} not in set(transformed_batch.keys())


@pytest.mark.parametrize(
    "random_batch",
    [{"batch_size": 64, "max_len": 200, "vocab_size": 3}],
    indirect=True,
)
def test_token_mask_transform(random_batch):
    transform = TokenMaskTransform(token_field="item_id_mask")
    transformed_batch = transform(random_batch)

    token_mask = transformed_batch["token_mask"]
    padding_mask = transformed_batch["item_id_mask"]
    not_padded_token_mask = token_mask[padding_mask]

    assert torch.any(not_padded_token_mask == 0)


@pytest.mark.parametrize(
    "mask_prob, batch",
    [
        (0.0, {"padding_mask": torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.bool)}),
        (1.0, {"padding_mask": torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.bool)}),
        (1e-6, {"padding_mask": torch.tensor([0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool)}),
    ],
)
def test_token_mask_transform_corner_cases(mask_prob, batch):
    transform = TokenMaskTransform(mask_prob=mask_prob, token_field="padding_mask")
    transformed_batch = transform(batch)

    token_mask = transformed_batch["token_mask"]
    padding_mask = batch["padding_mask"]
    not_padded_token_mask = token_mask[padding_mask]

    assert torch.any(not_padded_token_mask == 0)


@pytest.mark.parametrize("max_len", [1, 10])
def test_trim_transform(random_batch, max_len):
    features_to_trim = ["item_id", "cat_feature"]
    transform = TrimTransform(max_len, features_to_trim)
    transformed_batch = transform(random_batch)

    for feature in features_to_trim:
        assert transformed_batch[feature].shape[1] == max_len


def test_trim_transform_wrong_length(random_batch):
    features_to_trim = ["item_id", "cat_feature"]
    transform = TrimTransform(100, features_to_trim)
    with pytest.raises(AssertionError):
        transform(random_batch)


@pytest.mark.parametrize(
    "transform", [
        pytest.param(CopyTransform(mapping={"item_id_mask" : "padding_id"}), id="CopyTransform"),
        pytest.param(GroupTransform(mapping={"feature_tensors" : ["item_id"]}), id="GroupTransform"),
        pytest.param(NextTokenTransform(label_field="item_id", query_features=["user_id", "user_id_mask"]), id="NextTokenTransform"),
        pytest.param(RenameTransform(mapping={"item_id_mask" : "padding_id"}), id="RenameTransform"),
        pytest.param(SequenceRollTransform(field_name="item_id"), id="SequenceRollTransform"),
        pytest.param(TokenMaskTransform(token_field="item_id_mask"), id="TokenMaskTransform"),
        pytest.param(TrimTransform(seq_len=2, feature_names=["item_id"]), id="TrimTransform"),
        # pytest.param(UniformNegativeSamplingTransform(vocab_size=4, num_negative_samples=2), id="UniformNegativeSamplingTransform"),
        pytest.param(UnsqueezeTransform(column_name="item_id", dim=-1), id="UnsqueezeTransform"),
    ],
)
def test_immutability_input_batch(transform, random_batch):

    input_batch_id = id(random_batch)
    input_batch_keys = set(random_batch.keys())
    input_batch_items = set(random_batch.items())

    output_batch = transform(random_batch)

    assert id(output_batch) != input_batch_id

    assert id(random_batch) == input_batch_id
    assert set(random_batch.keys()) == input_batch_keys
    assert set(random_batch.items()) == input_batch_items
