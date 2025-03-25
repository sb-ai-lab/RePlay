from typing import NamedTuple, Optional, cast

import torch

from replay.data.nn import (
    MutableTensorMap,
    SequentialDataset,
    TensorMap,
)
from replay.models.nn.sequential.sasrec import SasRecTrainingDataset


class SasRecLLMTrainingBatch(NamedTuple):
    """
    Batch of data for training.
    Generated by `SasRecLLMTrainingDataset`.
    """

    user_profile_embeddings_batch: torch.Tensor
    existing_profile_binary_mask_batch: torch.BoolTensor
    query_id: torch.LongTensor
    padding_mask: torch.BoolTensor
    features: TensorMap
    labels: torch.LongTensor
    labels_padding_mask: torch.BoolTensor


class SasRecLLMTrainingDataset(SasRecTrainingDataset):
    """
    Dataset that generates samples to train SasRecLLM-like model
    """

    def __init__(
        self,
        sequential: SequentialDataset,
        max_sequence_length: int,
        user_profile_embeddings: torch.FloatTensor,
        existing_profile_binary_mask: torch.BoolTensor,
        sequence_shift: int = 1,
        sliding_window_step: Optional[None] = None,
        padding_value: int = 0,
        label_feature_name: Optional[str] = None,
    ) -> None:
        """
        :param sequential: Sequential dataset with training data.
        :param max_sequence_length: Max length of sequence.
        :param user_profile_embeddings: User profile embeddings tensor.
        :param existing_profile_binary_mask: Binary mask for missing profiles.
        :param sequence_shift: Shift of sequence to predict.
        :param sliding_window_step: A sliding window step.
            If not ``None`` provides iteration over sequences with window.
            Default: ``None``.
        :param padding_value: Value for padding a sequence to match the `max_sequence_length`.
            Default: ``0``.
        :param label_feature_name: Name of label feature in provided dataset.
            If ``None`` set an item_id_feature name from sequential dataset.
            Default: ``None``.
        """
        super().__init__(
            sequential=sequential,
            max_sequence_length=max_sequence_length,
            sequence_shift=sequence_shift,
            sliding_window_step=sliding_window_step,
            padding_value=padding_value,
            label_feature_name=label_feature_name,
        )

        self.user_profile_embeddings = user_profile_embeddings
        self.existing_profile_binary_mask = existing_profile_binary_mask

    def __getitem__(self, index: int) -> SasRecLLMTrainingBatch:
        query_id, padding_mask, features = self._inner[index]
        user_profile_emb_batch = self.user_profile_embeddings[query_id].squeeze(0)
        existing_profile_binary_mask_batch = self.existing_profile_binary_mask[query_id]

        assert self._label_feature_name
        labels = features[self._label_feature_name][self._sequence_shift :]
        labels_padding_mask = padding_mask[self._sequence_shift :]

        output_features: MutableTensorMap = {}
        for feature_name in self._schema:
            output_features[feature_name] = features[feature_name][: -self._sequence_shift]

        output_features_padding_mask = padding_mask[: -self._sequence_shift]

        return SasRecLLMTrainingBatch(
            user_profile_embeddings_batch=user_profile_emb_batch,
            existing_profile_binary_mask_batch=existing_profile_binary_mask_batch,
            query_id=query_id,
            features=output_features,
            padding_mask=cast(torch.BoolTensor, output_features_padding_mask),
            labels=cast(torch.LongTensor, labels),
            labels_padding_mask=cast(torch.BoolTensor, labels_padding_mask),
        )
