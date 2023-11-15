from abc import ABC, abstractmethod
from typing import Optional, Tuple

from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, PandasDataFrame, SparkDataFrame

if PYSPARK_AVAILABLE:
    from pyspark.sql import Window
    from pyspark.sql import functions as sf


SplitterReturnType = Tuple[DataFrameLike, DataFrameLike]


# pylint: disable=too-few-public-methods, too-many-instance-attributes
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

    # pylint: disable=too-many-arguments
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

    def __str__(self):
        return type(self).__name__

    # pylint: disable=too-many-arguments
    def _drop_cold_items_and_users(
        self,
        train: DataFrameLike,
        test: DataFrameLike,
    ) -> DataFrameLike:
        if isinstance(train, type(test)) is False:
            raise TypeError("Train and test dataframes must have consistent types")

        if isinstance(test, SparkDataFrame):
            return self._drop_cold_items_and_users_from_spark(train, test)

        return self._drop_cold_items_and_users_from_pandas(train, test)

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

        return self._recalculate_with_session_id_column_pandas(data)

    def _recalculate_with_session_id_column_pandas(self, data: PandasDataFrame) -> PandasDataFrame:
        agg_function_name = "first" if self.session_id_processing_strategy == "train" else "last"
        res = data.copy()
        res["is_test"] = res.groupby(
            [self.query_column, self.session_id_column]
        )["is_test"].transform(agg_function_name)

        return res

    def _recalculate_with_session_id_column_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        agg_function = sf.first if self.session_id_processing_strategy == "train" else sf.last
        res = data.withColumn(
            "is_test",
            agg_function("is_test").over(
                Window.orderBy(self.timestamp_column)
                .partitionBy(self.query_column, self.session_id_column)  # type: ignore
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            ),
        )

        return res
