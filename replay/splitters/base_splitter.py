import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import polars as pl

from replay.utils import (
    PYSPARK_AVAILABLE,
    DataFrameLike,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
)

if PYSPARK_AVAILABLE:
    from pyspark.sql import (
        Window,
        functions as sf,
    )


SplitterReturnType = Tuple[DataFrameLike, DataFrameLike]


class Splitter(ABC):
    """Base class"""

    _init_arg_names = [
        "drop_cold_users",
        "drop_cold_items",
        "query_column",
        "item_column",
        "timestamp_column",
        "session_id_column",
        "session_id_processing_strategy",
    ]

    def __init__(
        self,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        query_column: str = "query_id",
        item_column: Optional[str] = "item_id",
        timestamp_column: Optional[str] = "timestamp",
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param drop_cold_items: flag to remove items that are not in train data
        :param drop_cold_users: flag to remove users that are not in train data
        :param query_column: query id column name
        :param item_column: item id column name
        :param timestamp_column: timestamp column name
        :param session_id_column: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        self.drop_cold_users = drop_cold_users
        self.drop_cold_items = drop_cold_items
        self.query_column = query_column
        self.item_column = item_column
        self.timestamp_column = timestamp_column

        self.session_id_column = session_id_column
        self.session_id_processing_strategy = session_id_processing_strategy

    @property
    def _init_args(self):
        return {name: getattr(self, name) for name in self._init_arg_names}

    def save(self, path: str) -> None:
        """
        Method for saving splitter in `.replay` directory.
        """
        base_path = Path(path).with_suffix(".replay").resolve()
        base_path.mkdir(parents=True, exist_ok=True)

        splitter_dict = {}
        splitter_dict["init_args"] = self._init_args
        splitter_dict["_class_name"] = str(self)

        with open(base_path / "init_args.json", "w+") as file:
            json.dump(splitter_dict, file)

    @classmethod
    def load(cls, path: str) -> "Splitter":
        """
        Method for loading splitter from `.replay` directory.
        """
        base_path = Path(path).with_suffix(".replay").resolve()
        with open(base_path / "init_args.json", "r") as file:
            splitter_dict = json.loads(file.read())
        splitter = cls(**splitter_dict["init_args"])

        return splitter

    def __str__(self):
        return type(self).__name__

    def _drop_cold_items_and_users(
        self,
        train: DataFrameLike,
        test: DataFrameLike,
    ) -> DataFrameLike:
        if isinstance(train, type(test)) is False:
            msg = "Train and test dataframes must have consistent types"
            raise TypeError(msg)

        if isinstance(test, SparkDataFrame):
            return self._drop_cold_items_and_users_from_spark(train, test)
        if isinstance(test, PandasDataFrame):
            return self._drop_cold_items_and_users_from_pandas(train, test)
        else:
            return self._drop_cold_items_and_users_from_polars(train, test)

    def _drop_cold_items_and_users_from_pandas(
        self,
        train: PandasDataFrame,
        test: PandasDataFrame,
    ) -> PandasDataFrame:
        if self.drop_cold_items:
            test = test[test[self.item_column].isin(train[self.item_column])]

        if self.drop_cold_users:
            test = test[test[self.query_column].isin(train[self.query_column])]

        return test

    def _drop_cold_items_and_users_from_spark(
        self,
        train: SparkDataFrame,
        test: SparkDataFrame,
    ) -> SparkDataFrame:
        if self.drop_cold_items:
            train_tmp = train.select(sf.col(self.item_column).alias("item")).distinct()
            test = test.join(train_tmp, train_tmp["item"] == test[self.item_column]).drop("item")

        if self.drop_cold_users:
            train_tmp = train.select(sf.col(self.query_column).alias("user")).distinct()
            test = test.join(train_tmp, train_tmp["user"] == test[self.query_column]).drop("user")

        return test

    def _drop_cold_items_and_users_from_polars(
        self,
        train: PolarsDataFrame,
        test: PolarsDataFrame,
    ) -> PolarsDataFrame:
        if self.drop_cold_items:
            train_tmp = train.select(self.item_column).unique()
            test = test.join(train_tmp, on=self.item_column)

        if self.drop_cold_users:
            train_tmp = train.select(self.query_column).unique()
            test = test.join(train_tmp, on=self.query_column)

        return test

    @abstractmethod
    def _core_split(self, interactions: DataFrameLike) -> SplitterReturnType:
        """
        This method implements split strategy

        :param interactions: input DataFrame `[timestamp, user_id, item_id, relevance]`
        :returns: `train` and `test DataFrames
        """

    def split(self, interactions: DataFrameLike) -> SplitterReturnType:
        """
        Splits input DataFrame into train and test

        :param interactions: input DataFrame ``[timestamp, user_id, item_id, relevance]``
        :returns: List of splitted DataFrames
        """
        train, test = self._core_split(interactions)
        test = self._drop_cold_items_and_users(train, test)

        return train, test

    def _recalculate_with_session_id_column(self, data: DataFrameLike) -> DataFrameLike:
        if isinstance(data, SparkDataFrame):
            return self._recalculate_with_session_id_column_spark(data)
        if isinstance(data, PandasDataFrame):
            return self._recalculate_with_session_id_column_pandas(data)
        else:
            return self._recalculate_with_session_id_column_polars(data)

    def _recalculate_with_session_id_column_pandas(self, data: PandasDataFrame) -> PandasDataFrame:
        agg_function_name = "first" if self.session_id_processing_strategy == "train" else "last"
        res = data.copy()
        res["is_test"] = res.groupby([self.query_column, self.session_id_column])["is_test"].transform(
            agg_function_name
        )

        return res

    def _recalculate_with_session_id_column_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        agg_function = sf.first if self.session_id_processing_strategy == "train" else sf.last
        res = data.withColumn(
            "is_test",
            agg_function("is_test").over(
                Window.orderBy(self.timestamp_column)
                .partitionBy(self.query_column, self.session_id_column)
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            ),
        )

        return res

    def _recalculate_with_session_id_column_polars(self, data: PolarsDataFrame) -> PolarsDataFrame:
        agg_function = pl.Expr.first if self.session_id_processing_strategy == "train" else pl.Expr.last
        res = data.with_columns(
            agg_function(pl.col("is_test").sort_by(self.timestamp_column)).over(
                [self.query_column, self.session_id_column]
            )
        )

        return res
