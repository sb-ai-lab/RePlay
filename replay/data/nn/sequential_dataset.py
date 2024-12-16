import abc
import json
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame

from .schema import TensorSchema


class SequentialDataset(abc.ABC):
    """
    Abstract base class for sequential dataset
    """

    @abc.abstractmethod
    def __len__(self) -> int:  # pragma: no cover
        """
        Returns the length of the dataset.
        """

    @abc.abstractmethod
    def get_query_id(self, index: int) -> int:  # pragma: no cover
        """
        Getting a query id for a given index.

        :param index: the row number in the dataset.
        """

    @abc.abstractmethod
    def get_all_query_ids(self) -> np.ndarray:  # pragma: no cover
        """
        Getting a list of all query ids.
        """

    @abc.abstractmethod
    def get_sequence_length(self, index: int) -> int:  # pragma: no cover
        """
        Returns the length of the sequence at the specified index.

        :param index: the row number in the dataset.
        """

    @abc.abstractmethod
    def get_max_sequence_length(self) -> int:  # pragma: no cover
        """
        Returns the maximum length among all sequences from the `SequentialDataset`.
        """

    @abc.abstractmethod
    def get_sequence(self, index: Union[int, np.ndarray], feature_name: str) -> np.ndarray:  # pragma: no cover
        """
        Getting a sequence based on a given index and feature name.

        :param index: single index or list of indices.
        :param feature_name: the name of the feature.
        """

    @abc.abstractmethod
    def get_sequence_by_query_id(
        self, query_id: Union[int, np.ndarray], feature_name: str
    ) -> np.ndarray:  # pragma: no cover
        """
        Getting a sequence based on a given query id and feature name.

        :param query_id: single query id or list of query ids.
        :param feature_name: the name of the feature.
        """

    @abc.abstractmethod
    def filter_by_query_id(self, query_ids_to_keep: np.ndarray) -> "SequentialDataset":  # pragma: no cover
        """
        Returns a `SequentialDataset` that contains only query ids from the specified list.

        :param query_ids_to_keep: list of query ids.
        """

    @property
    @abc.abstractmethod
    def schema(self) -> TensorSchema:  # pragma: no cover
        """
        :returns: List of tensor features.
        """

    @staticmethod
    def keep_common_query_ids(
        lhs: "SequentialDataset", rhs: "SequentialDataset"
    ) -> Tuple["SequentialDataset", "SequentialDataset"]:
        """
        Returns `SequentialDatasets` that contain query ids from both datasets.

        :param lhs: `SequentialDataset`.
        :param rhs: `SequentialDataset`.
        """
        lhs_queries = lhs.get_all_query_ids()
        rhs_queries = rhs.get_all_query_ids()
        common_queries = np.intersect1d(lhs_queries, rhs_queries, assume_unique=True)
        lhs_filtered = lhs.filter_by_query_id(common_queries)
        rhs_filtered = rhs.filter_by_query_id(common_queries)
        return lhs_filtered, rhs_filtered

    def save(self, path: str) -> None:
        base_path = Path(path).with_suffix(".replay").resolve()
        base_path.mkdir(parents=True, exist_ok=True)

        sequential_dict = {}
        sequential_dict["_class_name"] = self.__class__.__name__
        self._sequences.reset_index().to_json(base_path / "sequences.json")
        sequential_dict["init_args"] = {
            "tensor_schema": self._tensor_schema._get_object_args(),
            "query_id_column": self._query_id_column,
            "item_id_column": self._item_id_column,
            "sequences_path": "sequences.json",
        }

        with open(base_path / "init_args.json", "w+") as file:
            json.dump(sequential_dict, file)


class PandasSequentialDataset(SequentialDataset):
    """
    Sequential dataset that stores sequences in PandasDataFrame format.
    """

    def __init__(
        self,
        tensor_schema: TensorSchema,
        query_id_column: str,
        item_id_column: str,
        sequences: PandasDataFrame,
    ) -> None:
        """
        :param tensor_schema: schema of tensor features.
        :param query_id_column: The name of the column containing query ids.
        :param item_id_column: The name of the column containing item ids.
        :param sequences: PandasDataFrame with sequences corresponding to the tensor schema.
        """
        self._check_if_schema_matches_data(tensor_schema, sequences)

        self._tensor_schema = tensor_schema
        self._query_id_column = query_id_column
        self._item_id_column = item_id_column

        if sequences.index.name != query_id_column:
            sequences = sequences.set_index(query_id_column)

        self._sequences = sequences

    def __len__(self) -> int:
        return len(self._sequences)

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
        for tensor_feature_name in tensor_schema:
            if tensor_feature_name not in data:
                msg = "Tensor schema does not match with provided data frame"
                raise ValueError(msg)

    @classmethod
    def load(cls, path: str, **kwargs) -> "PandasSequentialDataset":
        """
        Method for loading PandasSequentialDataset object from `.replay` directory.
        """
        base_path = Path(path).with_suffix(".replay").resolve()
        with open(base_path / "init_args.json", "r") as file:
            sequential_dict = json.loads(file.read())

        sequences = pd.read_json(base_path / sequential_dict["init_args"]["sequences_path"])
        dataset = cls(
            tensor_schema=TensorSchema._create_object_by_args(sequential_dict["init_args"]["tensor_schema"]),
            query_id_column=sequential_dict["init_args"]["query_id_column"],
            item_id_column=sequential_dict["init_args"]["item_id_column"],
            sequences=sequences,
        )

        return dataset


class PolarsSequentialDataset(PandasSequentialDataset):
    """
    Sequential dataset that stores sequences in PolarsDataFrame format.
    """

    def __init__(
        self,
        tensor_schema: TensorSchema,
        query_id_column: str,
        item_id_column: str,
        sequences: PolarsDataFrame,
    ) -> None:
        """
        :param tensor_schema: schema of tensor features.
        :param query_id_column: The name of the column containing query ids.
        :param item_id_column: The name of the column containing item ids.
        :param sequences: PolarsDataFrame with sequences corresponding to the tensor schema.
        """
        self._check_if_schema_matches_data(tensor_schema, sequences)

        self._tensor_schema = tensor_schema
        self._query_id_column = query_id_column
        self._item_id_column = item_id_column

        self._sequences = self._convert_polars_to_pandas(sequences)
        if self._sequences.index.name != query_id_column:
            self._sequences = self._sequences.set_index(query_id_column)

    def filter_by_query_id(self, query_ids_to_keep: np.ndarray) -> "PolarsSequentialDataset":
        filtered_sequences = self._sequences.loc[query_ids_to_keep]
        if filtered_sequences.index.name == self._query_id_column:
            filtered_sequences = filtered_sequences.reset_index()
        return PolarsSequentialDataset(
            tensor_schema=self._tensor_schema,
            query_id_column=self._query_id_column,
            item_id_column=self._item_id_column,
            sequences=self._convert_pandas_to_polars(filtered_sequences),
        )

    def _convert_polars_to_pandas(self, df: PolarsDataFrame) -> PandasDataFrame:
        pandas_df = PandasDataFrame(df.to_dict(as_series=False))

        for column in pandas_df.select_dtypes(include="object").columns:
            if isinstance(pandas_df[column].iloc[0], list):
                pandas_df[column] = pandas_df[column].apply(lambda x: np.array(x))

        return pandas_df

    def _convert_pandas_to_polars(self, df: PandasDataFrame) -> PolarsDataFrame:
        for column in df.select_dtypes(include="object").columns:
            if isinstance(df[column].iloc[0], np.ndarray):
                df[column] = df[column].apply(lambda x: x.tolist())

        return pl.from_dict(df.to_dict("list"))

    @classmethod
    def _check_if_schema_matches_data(cls, tensor_schema: TensorSchema, data: PolarsDataFrame) -> None:
        for tensor_feature_name in tensor_schema:
            if tensor_feature_name not in data:
                msg = "Tensor schema does not match with provided data frame"
                raise ValueError(msg)

    @classmethod
    def load(cls, path: str, **kwargs) -> "PandasSequentialDataset":
        """
        Method for loading PandasSequentialDataset object from `.replay` directory.
        """
        base_path = Path(path).with_suffix(".replay").resolve()
        with open(base_path / "init_args.json", "r") as file:
            sequential_dict = json.loads(file.read())

        sequences = pl.DataFrame(pd.read_json(base_path / sequential_dict["init_args"]["sequences_path"]))
        dataset = cls(
            tensor_schema=TensorSchema._create_object_by_args(sequential_dict["init_args"]["tensor_schema"]),
            query_id_column=sequential_dict["init_args"]["query_id_column"],
            item_id_column=sequential_dict["init_args"]["item_id_column"],
            sequences=sequences,
        )

        return dataset
