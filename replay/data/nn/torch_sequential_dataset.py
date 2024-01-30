from typing import Generator, NamedTuple, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from .schema import TensorFeatureInfo, TensorMap, TensorSchema
from .sequential_dataset import SequentialDataset


# We do not use dataclasses as PyTorch default collate
# function in dataloader supports only namedtuple
class TorchSequentialBatch(NamedTuple):
    """
    Batch of TorchSequentialDataset
    """
    query_id: torch.LongTensor
    padding_mask: torch.BoolTensor
    features: TensorMap


class TorchSequentialDataset(TorchDataset):
    """
    Torch dataset for sequential recommender models
    """

    def __init__(
        self,
        sequential: SequentialDataset,
        max_sequence_length: int,
        sliding_window_step: Optional[int] = None,
        padding_value: int = 0,
    ) -> None:
        """
        :param sequential: sequential dataset
        :param max_sequence_length: the maximum length of sequence
        :param sliding_window_step: value of offset from each sequence start during iteration,
            `None` means the offset will be equals to difference between actual sequence
            length and `max_sequence_length`.
            Default: `None`
        :param padding_value: value to pad sequences to desired length
        """
        super().__init__()
        self._sequential = sequential
        self._max_sequence_length = max_sequence_length
        self._sliding_window_step = sliding_window_step
        self._padding_value = padding_value
        self._index2sequence_map = self._build_index2sequence_map()

    def __len__(self) -> int:
        return len(self._index2sequence_map)

    def __getitem__(self, index: int) -> TorchSequentialBatch:
        sequence_index, sequence_offset = self._index2sequence_map[index]

        query_id = self._generate_query_id(sequence_index)
        padding_mask = self._generate_padding_mask(sequence_index)

        tensor_features = {
            feature_name: self._generate_tensor_feature(feature, sequence_index, sequence_offset)
            for feature_name, feature in self._sequential.schema.items()
        }

        return TorchSequentialBatch(
            query_id,
            padding_mask,
            tensor_features,
        )

    def _generate_query_id(self, sequence_index: int) -> torch.LongTensor:
        query_id = self._sequential.get_query_id(sequence_index)
        return torch.LongTensor([query_id])

    def _generate_padding_mask(self, sequence_index: int) -> torch.BoolTensor:
        mask = torch.ones(self._max_sequence_length, dtype=torch.bool)

        actual_sequence_len = self._sequential.get_sequence_length(sequence_index)
        if actual_sequence_len < self._max_sequence_length:
            mask[:-actual_sequence_len].fill_(0)

        return cast(torch.BoolTensor, mask)

    def _generate_tensor_feature(
        self,
        feature: TensorFeatureInfo,
        sequence_index: int,
        sequence_offset: int,
    ) -> torch.Tensor:
        sequence = self._sequential.get_sequence(sequence_index, feature.name)
        if feature.is_seq:
            sequence = sequence[sequence_offset : sequence_offset + self._max_sequence_length]  # noqa: E203

        tensor_dtype = self._get_tensor_dtype(feature)
        tensor_sequence = torch.tensor(sequence, dtype=tensor_dtype)
        if feature.is_seq:
            tensor_sequence = self._pad_sequence(tensor_sequence)

        return tensor_sequence

    def _pad_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        assert len(sequence) <= self._max_sequence_length
        if len(sequence) == self._max_sequence_length:
            return sequence

        # form shape for padded_sequence. Now supported one and two-dimentions features
        padded_sequence_shape: Union[Tuple[int, int], Tuple[int]]
        if len(sequence.shape) == 1:
            padded_sequence_shape = (self._max_sequence_length,)
        elif len(sequence.shape) == 2:
            padded_sequence_shape = (self._max_sequence_length, sequence.shape[1])
        else:
            raise ValueError(f"Unsupported shape for sequence: {len(sequence.shape)}")

        padded_sequence = torch.full(
            padded_sequence_shape,
            self._padding_value,
            dtype=sequence.dtype,
        )
        padded_sequence[-len(sequence) :].copy_(sequence)  # noqa: E203
        return padded_sequence

    def _get_tensor_dtype(self, feature: TensorFeatureInfo) -> torch.dtype:
        if feature.is_cat:
            return torch.long
        if feature.is_num:
            return torch.float32
        assert False, "Unknown tensor feature type"

    def _build_index2sequence_map(self) -> Sequence[Tuple[int, int]]:
        return list(self._iter_with_window())

    def _iter_with_window(self) -> Generator[Tuple[int, int], None, None]:
        for i in range(len(self._sequential)):
            actual_seq_len = self._sequential.get_sequence_length(i)
            left_seq_len = actual_seq_len - self._max_sequence_length

            if self._sliding_window_step is not None:
                offset_from_seq_beginning = left_seq_len
                while offset_from_seq_beginning > 0:
                    yield (i, offset_from_seq_beginning)
                    offset_from_seq_beginning -= self._sliding_window_step

                assert offset_from_seq_beginning <= 0
                yield (i, 0)
            else:
                offset_from_seq_beginning = max(0, left_seq_len)
                yield (i, offset_from_seq_beginning)


class TorchSequentialValidationBatch(NamedTuple):
    """
    Batch of TorchSequentialValidationDataset
    """
    query_id: torch.LongTensor
    padding_mask: torch.BoolTensor
    features: TensorMap
    ground_truth: torch.LongTensor
    train: torch.LongTensor


DEFAULT_GROUND_TRUTH_PADDING_VALUE = -1
DEFAULT_TRAIN_PADDING_VALUE = -2


class TorchSequentialValidationDataset(TorchDataset):
    """
    Torch dataset for sequential recommender models that additionally stores ground truth
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        sequential: SequentialDataset,
        ground_truth: SequentialDataset,
        train: SequentialDataset,
        max_sequence_length: int,
        padding_value: int = 0,
        sliding_window_step: Optional[int] = None,
        label_feature_name: Optional[str] = None,
    ):
        """
        :param sequential: validation sequential dataset
        :param ground_truth: validation ground_truth sequential dataset
        :param train: train sequential dataset
        :param max_sequence_length: the maximum length of sequence
        :param padding_value: value to pad sequences to desired length
        :param sliding_window_step: value of offset from each sequence start during iteration,
            `None` means the offset will be equals to difference between actual sequence
            length and `max_sequence_length`.
            Default: `None`
        :param label_feature_name: the name of the column containing the sequence of items.
        """
        self._check_if_schema_match(sequential.schema, ground_truth.schema)
        self._check_if_schema_match(sequential.schema, train.schema)

        if label_feature_name:
            if label_feature_name not in ground_truth.schema:
                raise ValueError("Label feature name not found in ground truth schema")

            if label_feature_name not in train.schema:
                raise ValueError("Label feature name not found in train schema")

            if not ground_truth.schema[label_feature_name].is_cat:
                raise ValueError("Label feature must be categorical")

            if not ground_truth.schema[label_feature_name].is_seq:
                raise ValueError("Label feature must be sequential")

        if len(np.intersect1d(sequential.get_all_query_ids(), ground_truth.get_all_query_ids())) == 0:
            raise ValueError("Sequential data and ground truth must contain the same query IDs")

        self._ground_truth = ground_truth
        self._train = train
        self._item_count = ground_truth.schema.item_id_features.item().cardinality
        self._label_feature_name = label_feature_name or ground_truth.schema.item_id_feature_name
        self._max_ground_truth_length = ground_truth.get_max_sequence_length()
        self._max_train_length = train.get_max_sequence_length()

        self._inner = TorchSequentialDataset(
            sequential=sequential,
            max_sequence_length=max_sequence_length,
            sliding_window_step=sliding_window_step,
            padding_value=padding_value,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> TorchSequentialValidationBatch:
        query_id, padding_mask, features = self._inner[index]

        return TorchSequentialValidationBatch(
            query_id=query_id,
            padding_mask=padding_mask,
            features=features,
            ground_truth=self._get_ground_truth(int(query_id.item())),
            train=self._get_train(int(query_id.item())),
        )

    def _get_ground_truth(self, query_id: int) -> torch.LongTensor:
        assert self._label_feature_name

        ground_truth_sequence = self._ground_truth.get_sequence_by_query_id(
            query_id,
            self._label_feature_name,
        )

        placeholder = np.full(self._max_ground_truth_length, DEFAULT_GROUND_TRUTH_PADDING_VALUE, dtype=np.int64)
        np.copyto(placeholder[: len(ground_truth_sequence)], ground_truth_sequence)
        return torch.LongTensor(placeholder)

    def _get_train(self, query_id: int) -> torch.LongTensor:
        assert self._label_feature_name

        train_sequence = self._train.get_sequence_by_query_id(
            query_id,
            self._label_feature_name,
        )

        placeholder = np.full(self._max_train_length, DEFAULT_TRAIN_PADDING_VALUE, dtype=np.int64)
        np.copyto(placeholder[: len(train_sequence)], train_sequence)
        return torch.LongTensor(placeholder)

    @classmethod
    def _check_if_schema_match(
        cls,
        sequential_schema: TensorSchema,
        ground_truth_schema: TensorSchema,
    ) -> None:
        sequential_item_feature = sequential_schema.item_id_features.item()
        ground_truth_item_feature = ground_truth_schema.item_id_features.item()

        if sequential_item_feature.name != ground_truth_item_feature.name:
            raise ValueError("Schema mismatch: item feature name does not match ground truth")

        if sequential_item_feature.cardinality != ground_truth_item_feature.cardinality:
            raise ValueError("Schema mismatch: item feature cardinality does not match ground truth")
