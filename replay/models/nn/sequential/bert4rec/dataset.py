import abc
from typing import NamedTuple, Optional, Tuple, cast

import torch
from torch.utils.data import Dataset as TorchDataset

from replay.data.nn import (
    MutableTensorMap,
    SequentialDataset,
    TensorMap,
    TensorSchema,
    TorchSequentialDataset,
    TorchSequentialValidationDataset,
)


class BertTrainingBatch(NamedTuple):
    """
    Batch of data for training.
    Generated by `BertTrainingDataset`.
    """

    query_id: torch.LongTensor
    padding_mask: torch.BoolTensor
    features: TensorMap
    tokens_mask: torch.BoolTensor
    labels: torch.LongTensor


# pylint: disable=too-few-public-methods
class BertMasker(abc.ABC):
    """
    Interface for a token masking strategy during BERT model training
    """

    @abc.abstractmethod
    def mask(self, paddings: torch.BoolTensor) -> torch.BoolTensor:  # pragma: no cover
        """
        Mask random tokens for only not padded tokens.

        :param paddings: Padding mask where ``0`` is <PAD> and ``1`` otherwise.

        :returns: Mask of sequence where ``0`` is masked and ``1`` otherwise.
        """


# pylint: disable=too-few-public-methods
class UniformBertMasker(BertMasker):
    """
    Token masking strategy that mask random token with uniform distribution.
    """

    def __init__(self, mask_prob: float = 0.15, generator: Optional[torch.Generator] = None) -> None:
        """
        :param mask_prob: Probability of masking each token in sequence.
            Default: ``0.15``.
        :param generator: A pseudorandom number generator for sampling.
            Default: ``None``.
        """
        super().__init__()
        self.generator = generator
        self.mask_prob = mask_prob

    def mask(self, paddings: torch.BoolTensor) -> torch.BoolTensor:
        """
        Mask random token with uniform distribution for only not padded tokens.

        :param paddings: Padding mask where ``0`` is <PAD> and ``1`` otherwise.

        :returns: Mask of sequence where ``0`` is masked and ``1`` otherwise.
        """
        mask_prob = torch.rand(paddings.size(-1), dtype=torch.float32, generator=self.generator)

        # mask[i], 0 ~ mask_prob, 1 ~ (1 - mask_prob)
        mask = (mask_prob * paddings) >= self.mask_prob

        # Fix corner cases in mask
        # 1. If all token are not masked, add mask to the end
        if mask.all():
            mask[-1] = 0
        # 2. If all token are masked, add non-masked before the last
        elif (not mask.any()) and (len(mask) > 1):
            mask[-2] = 1

        return cast(torch.BoolTensor, mask)


class BertTrainingDataset(TorchDataset):
    """
    Dataset that generates samples to train BERT-like model
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        sequential: SequentialDataset,
        max_sequence_length: int,
        mask_prob: float = 0.15,
        sliding_window_step: Optional[int] = None,
        label_feature_name: Optional[str] = None,
        custom_masker: Optional[BertMasker] = None,
        padding_value: int = 0,
    ) -> None:
        """
        :param sequential: Sequential dataset with training data.
        :param max_sequence_length: Max length of sequence.
        :param mask_prob: Probability of masking each token in sequence.
            Default: ``0.15``.
        :param sliding_window_step: A sliding window step.
            If not ``None`` provides iteration over sequences with window.
            Default: ``None``.
        :param label_feature_name: Name of label feature in provided dataset.
            If ``None`` set an item_id_feature name from sequential dataset.
            Default: ``None``.
        :param custom_masker: Masker object to generate masks for Bert training.
            If ``None`` set a UniformBertMasker with provided `mask_prob`.
            Default: ``None``.
        :param padding_value: Value for padding a sequence to match the `max_sequence_length`.
            Default: ``0``.
        """
        super().__init__()
        if label_feature_name:
            if label_feature_name not in sequential.schema:
                raise ValueError("Label feature name not found in provided schema")

            if not sequential.schema[label_feature_name].is_cat:
                raise ValueError("Label feature must be categorical")

            if not sequential.schema[label_feature_name].is_seq:
                raise ValueError("Label feature must be sequential")

        self._max_sequence_length = max_sequence_length
        self._label_feature_name = label_feature_name or sequential.schema.item_id_feature_name
        self._masker = custom_masker or UniformBertMasker(mask_prob)

        self._inner = TorchSequentialDataset(
            sequential=sequential,
            max_sequence_length=self._max_sequence_length,
            sliding_window_step=sliding_window_step,
            padding_value=padding_value,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> BertTrainingBatch:
        query_id, padding_mask, features = self._inner[index]
        tokens_mask = self._masker.mask(padding_mask)

        assert self._label_feature_name
        labels = features[self._label_feature_name]

        return BertTrainingBatch(
            query_id=query_id,
            padding_mask=padding_mask,
            features=features,
            tokens_mask=tokens_mask,
            labels=cast(torch.LongTensor, labels),
        )


class BertPredictionBatch(NamedTuple):
    """
    Batch of data for model inference.
    Generated by `BertPredictionDataset`.
    """

    query_id: torch.LongTensor
    padding_mask: torch.BoolTensor
    features: TensorMap
    tokens_mask: torch.BoolTensor


class BertPredictionDataset(TorchDataset):
    """
    Dataset that generates samples to infer BERT-like model
    """

    def __init__(
        self,
        sequential: SequentialDataset,
        max_sequence_length: int,
        padding_value: int = 0,
    ) -> None:
        """
        :param sequential: Sequential dataset with data to make predictions at.
        :param max_sequence_length: Max length of sequence.
        :param padding_value: Value for padding a sequence to match the `max_sequence_length`.
            Default: ``0``.
        """
        self._schema = sequential.schema
        self._inner = TorchSequentialDataset(
            sequential=sequential,
            max_sequence_length=max_sequence_length,
            padding_value=padding_value,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> BertPredictionBatch:
        query_id, padding_mask, features = self._inner[index]

        shifted_features, shifted_padding_mask, tokens_mask = _shift_features(self._schema, features, padding_mask)

        return BertPredictionBatch(
            query_id=query_id,
            padding_mask=shifted_padding_mask,
            features=shifted_features,
            tokens_mask=tokens_mask,
        )


class BertValidationBatch(NamedTuple):
    """
    Batch of data for validation.
    Generated by `BertValidationDataset`.
    """

    query_id: torch.LongTensor
    padding_mask: torch.BoolTensor
    features: TensorMap
    tokens_mask: torch.BoolTensor
    ground_truth: torch.LongTensor
    train: torch.LongTensor


class BertValidationDataset(TorchDataset):
    """
    Dataset that generates samples to infer and validate BERT-like model
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        sequential: SequentialDataset,
        ground_truth: SequentialDataset,
        train: SequentialDataset,
        max_sequence_length: int,
        padding_value: int = 0,
        label_feature_name: Optional[str] = None,
    ):
        """
        :param sequential: Sequential dataset with data to make predictions at.
        :param ground_truth: Sequential dataset with ground truth predictions.
        :param train: Sequential dataset with training data.
        :param max_sequence_length: Max length of sequence.
        :param padding_value: Value for padding a sequence to match the `max_sequence_length`.
            Default: ``0``.
        :param label_feature_name: Name of label feature in provided dataset.
            If ``None`` set an item_id_feature name from sequential dataset.
            Default: ``None``.
        """
        self._schema = sequential.schema
        self._inner = TorchSequentialValidationDataset(
            sequential=sequential,
            ground_truth=ground_truth,
            max_sequence_length=max_sequence_length,
            padding_value=padding_value,
            label_feature_name=label_feature_name,
            train=train,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> BertValidationBatch:
        query_id, padding_mask, features, ground_truth, train = self._inner[index]

        shifted_features, shifted_padding_mask, tokens_mask = _shift_features(self._schema, features, padding_mask)

        return BertValidationBatch(
            query_id=query_id,
            padding_mask=shifted_padding_mask,
            features=shifted_features,
            tokens_mask=tokens_mask,
            ground_truth=ground_truth,
            train=train,
        )


def _shift_features(
    schema: TensorSchema,
    features: TensorMap,
    padding_mask: torch.BoolTensor,
) -> Tuple[TensorMap, torch.BoolTensor, torch.BoolTensor]:
    shifted_features: MutableTensorMap = {}
    for feature_name, feature in schema.items():
        if feature.is_seq:
            shifted_features[feature_name] = _shift_seq(features[feature_name])
        else:
            shifted_features[feature_name] = features[feature_name]

    # [0, 0, 1, 1, 1] -> [0, 1, 1, 1, 0]
    tokens_mask = _shift_seq(padding_mask)

    # [0, 1, 1, 1, 0] -> [0, 1, 1, 1, 1]
    shifted_padding_mask = tokens_mask.clone()
    shifted_padding_mask[-1] = 1

    return (
        shifted_features,
        cast(torch.BoolTensor, shifted_padding_mask),
        cast(torch.BoolTensor, tokens_mask),
    )


def _shift_seq(seq: torch.Tensor) -> torch.Tensor:
    shifted_seq = seq.roll(-1, dims=0)
    shifted_seq[-1, ...] = 0
    return shifted_seq
