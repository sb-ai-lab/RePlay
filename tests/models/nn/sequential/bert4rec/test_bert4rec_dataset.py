import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.sequential.bert4rec import (
        BertPredictionDataset,
        BertTrainingDataset,
        BertValidationDataset,
        UniformBertMasker,
    )

torch = pytest.importorskip("torch")


@pytest.mark.torch
@pytest.mark.parametrize(
    "mask_prob, padding_mask, result",
    [
        (
            0.0,
            torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.bool),
            torch.tensor([1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.bool),
        ),
        (
            1.0,
            torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.bool),
            torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool),
        ),
        (
            1e-6,
            torch.tensor([0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool),
            torch.tensor([0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool),
        ),
    ],
)
def test_uniform_bert_masking_corner_cases(mask_prob, padding_mask, result):
    masker = UniformBertMasker(mask_prob=mask_prob)
    tokens_mask = masker.mask(paddings=padding_mask)

    assert all(tokens_mask == result)


@pytest.mark.torch
@pytest.mark.parametrize(
    "max_len, feature_name, exception, exception_text",
    [
        (8, "fake_user_id", pytest.raises(ValueError), "Label feature name not found in provided schema"),
        (8, "some_item_feature", pytest.raises(ValueError), "Label feature must be categorical"),
        (8, "some_user_feature", pytest.raises(ValueError), "Label feature must be sequential"),
    ],
)
def test_bert_training_dataset_exceptions(wrong_sequential_dataset, max_len, feature_name, exception, exception_text):
    with exception as exc:
        BertTrainingDataset(wrong_sequential_dataset, max_len, label_feature_name=feature_name)

    assert str(exc.value) == exception_text


@pytest.mark.torch
def test_bert_datasets_length(sequential_dataset):
    assert len(BertTrainingDataset(sequential_dataset, 8)) == 4
    assert len(BertPredictionDataset(sequential_dataset, 8)) == 4
    assert len(BertValidationDataset(sequential_dataset, sequential_dataset, sequential_dataset, 8)) == 4


@pytest.mark.torch
def test_bert_training_dataset_getitem(sequential_dataset):
    batch = BertTrainingDataset(
        sequential_dataset,
        max_sequence_length=8,
        label_feature_name="some_item_feature",
        padding_value=-1,
        mask_prob=0.0,
    )[0]

    assert batch.query_id.item() == 0
    assert all(batch.padding_mask == torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool))
    assert all(batch.tokens_mask == torch.tensor([1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.bool))
    assert all(batch.labels == torch.tensor([-1, -1, -1, -1, -1, -1, 1, 2]))


@pytest.mark.torch
def test_bert_prediction_dataset_getitem(sequential_dataset):
    batch = BertPredictionDataset(sequential_dataset, 8, padding_value=-1)[1]

    assert batch.query_id.item() == 1
    assert all(batch.padding_mask == torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool))
    assert all(batch.tokens_mask == torch.tensor([0, 0, 0, 0, 1, 1, 1, 0], dtype=torch.bool))


@pytest.mark.torch
def test_bert_validation_dataset_getitem(sequential_dataset):
    batch = BertValidationDataset(sequential_dataset, sequential_dataset, sequential_dataset, 8)[2]

    assert batch.query_id.item() == 2
    assert all(batch.padding_mask == torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool))
    assert all(batch.tokens_mask == torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool))
    assert all(batch.ground_truth == torch.tensor([1, -1, -1, -1, -1, -1]))
    assert all(batch.train == torch.tensor([1, -2, -2, -2, -2, -2]))
