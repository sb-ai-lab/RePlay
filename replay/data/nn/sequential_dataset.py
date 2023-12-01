import abc
from typing import Tuple, Union

import numpy as np
from pandas import DataFrame as PandasDataFrame

from replay.data.schema import FeatureType
from replay.data.nn.schema import TensorSchema


# pylint: disable=missing-function-docstring
class SequentialDataset(abc.ABC):
    """
    Abstract base class for sequential dataset
    """

    @abc.abstractmethod
    def __len__(self) -> int:  # pragma: no cover
        pass

    @abc.abstractmethod
    def get_query_id(self, index: int) -> int:  # pragma: no cover
        pass

    @abc.abstractmethod
    def get_all_query_ids(self) -> np.ndarray:  # pragma: no cover
        pass

    @abc.abstractmethod
    def get_sequence_length(self, index: int) -> int:  # pragma: no cover
        pass

    @abc.abstractmethod
    def get_max_sequence_length(self) -> int:  # pragma: no cover
        pass

    @abc.abstractmethod
    def get_sequence(self, index: Union[int, np.ndarray], feature_name: str) -> np.ndarray:  # pragma: no cover
        pass

    @abc.abstractmethod
    def get_sequence_by_query_id(
        self, query_id: Union[int, np.ndarray], feature_name: str
    ) -> np.ndarray:  # pragma: no cover
        pass

    @abc.abstractmethod
    def filter_by_query_id(self, query_ids_to_keep: np.ndarray) -> "SequentialDataset":  # pragma: no cover
        pass

    @property
    @abc.abstractmethod
    def schema(self) -> TensorSchema:  # pragma: no cover
        pass

    @staticmethod
    def keep_common_query_ids(
        lhs: "SequentialDataset", rhs: "SequentialDataset"
    ) -> Tuple["SequentialDataset", "SequentialDataset"]:
        lhs_queries = lhs.get_all_query_ids()
        rhs_queries = rhs.get_all_query_ids()
        common_queries = np.intersect1d(lhs_queries, rhs_queries, assume_unique=True)
        lhs_filtered = lhs.filter_by_query_id(common_queries)
        rhs_filtered = rhs.filter_by_query_id(common_queries)
        return lhs_filtered, rhs_filtered


class PandasSequentialDataset(SequentialDataset):
    """
    Sequential dataset that stores data in Pandas
    """

    def __init__(
        self,
        tensor_schema: TensorSchema,
        query_id_column: str,
        item_id_column: str,
        sequences: PandasDataFrame,
    ) -> None:
        self._check_if_schema_matches_data(tensor_schema, sequences)

        self._tensor_schema = tensor_schema
        self._query_id_column = query_id_column
        self._item_id_column = item_id_column

        if sequences.index.name != query_id_column:
            sequences = sequences.set_index(query_id_column)

        self._sequences = sequences

        for feature in tensor_schema.all_features:
            if feature.feature_type == FeatureType.CATEGORICAL:
                # pylint: disable=protected-access
                feature._set_cardinality_callback(self.cardinality_callback)

    def __len__(self) -> int:
        return len(self._sequences)

    def cardinality_callback(self, column: str) -> int:
        if self._query_id_column == column:
            return self._sequences.index.nunique()
        return len({x for seq in self._sequences[column] for x in seq})

    def get_query_id(self, index: int) -> int:
        return self._sequences.index[index]

    def get_all_query_ids(self) -> np.ndarray:
        return self._sequences.index.values

    def get_sequence_length(self, index: int) -> int:
        return len(self._sequences[self._item_id_column].iloc[index])

    def get_max_sequence_length(self) -> int:
        return max(len(seq) for seq in self._sequences[self._item_id_column])

    def get_sequence(self, index: Union[int, np.ndarray], feature_name: str) -> np.ndarray:
        return np.array(self._sequences[feature_name].iloc[index])

    def get_sequence_by_query_id(self, query_id: Union[int, np.ndarray], feature_name: str) -> np.ndarray:
        try:
            return np.array(self._sequences[feature_name].loc[query_id])
        except KeyError:
            return np.array([], dtype=np.int64)

    def filter_by_query_id(self, query_ids_to_keep: np.ndarray) -> "PandasSequentialDataset":
        filtered_sequences = self._sequences.loc[query_ids_to_keep]
        return PandasSequentialDataset(
            tensor_schema=self._tensor_schema,
            query_id_column=self._query_id_column,
            item_id_column=self._item_id_column,
            sequences=filtered_sequences,
        )

    @property
    def schema(self) -> TensorSchema:
        return self._tensor_schema

    @classmethod
    def _check_if_schema_matches_data(cls, tensor_schema: TensorSchema, data: PandasDataFrame) -> None:
        for tensor_feature_name in tensor_schema.keys():
            if tensor_feature_name not in data:
                raise ValueError("Tensor schema does not match with provided data frame")
