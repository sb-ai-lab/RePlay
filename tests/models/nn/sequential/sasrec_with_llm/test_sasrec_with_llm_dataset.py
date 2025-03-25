import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.sequential.sasrec_with_llm import SasRecLLMTrainingDataset

torch = pytest.importorskip("torch")


@pytest.mark.torch
@pytest.mark.parametrize(
    "max_len, feature_name, exception, exception_text",
    [
        (8, "fake_user_id", pytest.raises(ValueError), "Label feature name not found in provided schema"),
        (8, "some_item_feature", pytest.raises(ValueError), "Label feature must be categorical"),
        (8, "some_user_feature", pytest.raises(ValueError), "Label feature must be sequential"),
    ],
)
def test_sasrec_training_dataset_exceptions(wrong_sequential_dataset,
                                            max_len,
                                            feature_name,
                                            exception,
                                            exception_text,
                                            user_profile_embeddings,
                                            profile_binary_mask_getter):
    with exception as exc:
        existing_profile_binary_mask = profile_binary_mask_getter(wrong_sequential_dataset, user_profile_embeddings)
        SasRecLLMTrainingDataset(sequential=wrong_sequential_dataset,
                                 max_sequence_length=max_len,
                                 user_profile_embeddings=user_profile_embeddings,
                                 existing_profile_binary_mask=existing_profile_binary_mask,
                                 label_feature_name=feature_name)

    assert str(exc.value) == exception_text


@pytest.mark.torch
def test_sasrec_datasets_length(sequential_dataset, user_profile_embeddings, profile_binary_mask_getter):
    existing_profile_binary_mask = profile_binary_mask_getter(sequential_dataset, user_profile_embeddings)
    assert len(SasRecLLMTrainingDataset(sequential=sequential_dataset,
                                        max_sequence_length=8,
                                        user_profile_embeddings=user_profile_embeddings,
                                        existing_profile_binary_mask=existing_profile_binary_mask,
                                        )) == 4


@pytest.mark.torch
def test_sasrec_training_dataset_getitem(sequential_dataset, user_profile_embeddings, profile_binary_mask_getter):
    existing_profile_binary_mask = profile_binary_mask_getter(sequential_dataset, user_profile_embeddings)
    batch = SasRecLLMTrainingDataset(sequential=sequential_dataset,
                                     max_sequence_length=8,
                                     user_profile_embeddings=user_profile_embeddings,
                                     existing_profile_binary_mask=existing_profile_binary_mask,
                                     label_feature_name="item_id")[0]

    assert batch.query_id.item() == 0
    assert all(batch.padding_mask == torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool))
    assert all(batch.labels_padding_mask == torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool))
    assert all(batch.labels == torch.tensor([-1, -1, -1, -1, -1, -1, 0, 1], dtype=torch.long))

