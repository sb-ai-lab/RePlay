import copy

import pytest
import torch

from replay.nn.transform import (
    CopyTransform,
    GroupTransform,
    MultiClassNegativeSamplingTransform,
    NextTokenTransform,
    RenameTransform,
    SelectTransform,
    SequenceRollTransform,
    TokenMaskTransform,
    TrimTransform,
    UniformNegativeSamplingTransform,
    UnsqueezeTransform,
)


@pytest.mark.parametrize(
    "random_batch",
    [
        ({"batch_size": 5, "max_len": 10, "cardinality": 30}),
        ({"batch_size": 64, "max_len": 200, "cardinality": 3}),
        ({"batch_size": 1024, "max_len": 500, "cardinality": 300}),
    ],
    indirect=True,
)
@pytest.mark.parametrize("shift", [1, 5])
def test_next_token_label_transform(random_batch, shift):
    label_field = "item_id"
    ignore_features = ["user_id", "user_id_mask", "negative_selector"]
    transform = NextTokenTransform(label_field=label_field, ignore_features=ignore_features, shift=shift)
    transformed_batch = transform(random_batch)

    for feature in random_batch.keys():
        if any(feature.startswith(q) for q in ignore_features):
            torch.testing.assert_close(transformed_batch[feature], random_batch[feature])
        else:
            torch.testing.assert_close(transformed_batch[feature], random_batch[feature][:, :-shift])

    torch.testing.assert_close(transformed_batch["positive_labels"], random_batch[label_field][:, shift:])
    torch.testing.assert_close(
        transformed_batch["positive_labels_mask"], random_batch[f"{label_field}_mask"][:, shift:]
    )


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


def test_rename_transform(random_batch):
    transform = RenameTransform({"user_id": "query_id", "item_id_mask": "padding_mask"})

    source_batch = copy.deepcopy(random_batch)
    transformed_batch = transform(random_batch)

    assert transformed_batch["query_id"].equal(source_batch["user_id"])
    assert transformed_batch["padding_mask"].equal(source_batch["item_id_mask"])

    assert {"query_id", "padding_mask"}.isdisjoint(set(source_batch.keys()))
    assert {"user_id", "item_id_mask"}.isdisjoint(set(transformed_batch.keys()))


@pytest.mark.parametrize(
    "random_batch, cardinality, num_negative_samples",
    [
        ({"batch_size": 5, "max_len": 10, "cardinality": 30}, 30, 29),
        ({"batch_size": 64, "max_len": 200, "cardinality": 3}, 3, 1),
    ],
    indirect=["random_batch"],
)
@pytest.mark.parametrize("generate_sample_distribution", [True, False])
def test_negative_sampling_transform(random_batch, cardinality, num_negative_samples, generate_sample_distribution):
    sample_distribution = None
    if generate_sample_distribution:
        sample_distribution = torch.rand(cardinality)

    transform = UniformNegativeSamplingTransform(
        cardinality=cardinality, num_negative_samples=num_negative_samples, sample_distribution=sample_distribution
    )

    transformed_batch = transform(random_batch)

    assert "negative_labels" in transformed_batch.keys()
    assert transformed_batch["negative_labels"].size() == (num_negative_samples,)


@pytest.mark.parametrize(
    "random_batch, cardinality, num_classes, num_negative_samples",
    [
        ({"batch_size": 5, "max_len": 10, "cardinality": 30, "num_classes": 3}, 30, 3, 5),
        ({"batch_size": 64, "max_len": 200, "cardinality": 3, "num_classes": 2}, 3, 2, 1),
    ],
    indirect=["random_batch"],
)
def test_multiclass_negative_sampling_transform(random_batch, cardinality, num_classes, num_negative_samples):
    batch_size = random_batch["negative_selector"].size(0)

    generator = torch.Generator().manual_seed(0)
    sample_mask = torch.nn.functional.one_hot(torch.randint(0, num_classes, (cardinality,), generator=generator)).T

    transform = MultiClassNegativeSamplingTransform(
        num_negative_samples=num_negative_samples, sample_mask=sample_mask, generator=generator
    )

    transformed_batch = transform(random_batch)

    assert "negative_labels" in transformed_batch.keys()
    assert transformed_batch["negative_labels"].size() == (batch_size, num_negative_samples)
    for i in range(batch_size):
        class_i = transformed_batch["negative_selector"][i]
        all_class_ids = torch.where(sample_mask[class_i] > 0)[0]
        generated_negatives = transformed_batch["negative_labels"][i]
        assert torch.isin(generated_negatives, all_class_ids).all()


@pytest.mark.parametrize(
    "cardinality, num_negative_samples, sample_distribution",
    [(10, 1, torch.rand(100)), (10, 100, None)],
    ids=["Incorrect sample distribution last shape", "Incorrect num_negative_samples"],
)
def test_negative_sampling_raises(cardinality, num_negative_samples, sample_distribution):
    with pytest.raises(ValueError):
        UniformNegativeSamplingTransform(
            cardinality=cardinality, num_negative_samples=num_negative_samples, sample_distribution=sample_distribution
        )


@pytest.mark.parametrize(
    "num_negative_samples, sample_mask",
    [
        (100, torch.rand(5)),
        (100, torch.rand((2, 100))),
    ],
    ids=["Incorrect sample mask dim", "Incorrect num_negative_samples"],
)
def test_multiclass_negative_sampling_raises(num_negative_samples, sample_mask):
    with pytest.raises(ValueError):
        MultiClassNegativeSamplingTransform(num_negative_samples=num_negative_samples, sample_mask=sample_mask)


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
    [{"batch_size": 64, "max_len": 200, "cardinality": 3}],
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
    "transform",
    [
        pytest.param(CopyTransform(mapping={"item_id_mask": "padding_id"}), id="CopyTransform"),
        pytest.param(GroupTransform(mapping={"feature_tensors": ["item_id"]}), id="GroupTransform"),
        pytest.param(
            NextTokenTransform(label_field="item_id", ignore_features=["user_id", "user_id_mask"]),
            id="NextTokenTransform",
        ),
        pytest.param(RenameTransform(mapping={"item_id_mask": "padding_id"}), id="RenameTransform"),
        pytest.param(SelectTransform(["item_id"]), id="SequenceRollTransform"),
        pytest.param(SequenceRollTransform(field_name="item_id"), id="SequenceRollTransform"),
        pytest.param(TokenMaskTransform(token_field="item_id_mask"), id="TokenMaskTransform"),
        pytest.param(TrimTransform(seq_len=2, feature_names=["item_id"]), id="TrimTransform"),
        pytest.param(
            UniformNegativeSamplingTransform(cardinality=4, num_negative_samples=2),
            id="UniformNegativeSamplingTransform",
        ),
        pytest.param(
            MultiClassNegativeSamplingTransform(num_negative_samples=2, sample_mask=torch.rand((3, 4))),
            id="MultiClassNegativeSamplingTransform",
        ),
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


def test_select_transform(random_batch):
    features = ["item_id", "cat_feature"]
    transform = SelectTransform(features)
    transformed_batch = transform(random_batch)

    assert set(features) == set(transformed_batch.keys())
