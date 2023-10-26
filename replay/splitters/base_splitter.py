from abc import ABC, abstractmethod
from typing import Optional, Tuple

from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as sf, Window

from replay.data import AnyDataFrame

SplitterReturnType = Tuple[AnyDataFrame, AnyDataFrame]


# pylint: disable=too-few-public-methods, too-many-instance-attributes
class Splitter(ABC):
    """Base class"""

    _init_arg_names = [
        "drop_cold_users",
        "drop_cold_items",
        "user_col",
        "item_col",
        "timestamp_col",
        "session_id_col",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        drop_cold_items: bool,
        drop_cold_users: bool,
        user_col: str = "user_idx",
        item_col: Optional[str] = "item_idx",
        timestamp_col: Optional[str] = "timestamp",
        rating_col: Optional[str] = "relevance",
        session_id_col: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param drop_cold_items: flag to remove items that are not in train data
        :param drop_cold_users: flag to remove users that are not in train data
        :param user_col: user id column name
        :param item_col: item id column name
        :param timestamp_col: timestamp column name
        :param rating_col: rating column name
        :param session_id_col: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        self.drop_cold_users = drop_cold_users
        self.drop_cold_items = drop_cold_items
        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col
        self.rating_col = rating_col

        self.session_id_col = session_id_col
        self.session_id_processing_strategy = session_id_processing_strategy

        if session_id_processing_strategy not in ["train", "test"]:
            raise NotImplementedError("session_id_processing_strategy can only be: 'train' or 'test'.")

    @abstractmethod
    def _get_order_of_sort(self) -> list:  # pragma: no cover
        pass

    @property
    def _init_args(self):
        return {name: getattr(self, name) for name in self._init_arg_names}

    def __str__(self):
        return type(self).__name__

    # pylint: disable=too-many-arguments
    def _drop_cold_items_and_users(
        self,
        train: AnyDataFrame,
        test: AnyDataFrame,
    ) -> AnyDataFrame:
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
            test = test[test[self.item_col].isin(train[self.item_col])]

        if self.drop_cold_users:
            test = test[test[self.user_col].isin(train[self.user_col])]

        if self.drop_cold_users or self.drop_cold_items:
            test = test.sort_values(self._get_order_of_sort())

        return test

    def _drop_cold_items_and_users_from_spark(
        self,
        train: SparkDataFrame,
        test: SparkDataFrame,
    ) -> SparkDataFrame:

        if self.drop_cold_items:
            train_tmp = train.select(sf.col(self.item_col).alias("item")).distinct()
            test = test.join(train_tmp, train_tmp["item"] == test[self.item_col]).drop("item")

        if self.drop_cold_users:
            train_tmp = train.select(sf.col(self.user_col).alias("user")).distinct()
            test = test.join(train_tmp, train_tmp["user"] == test[self.user_col]).drop("user")

        if self.drop_cold_users or self.drop_cold_items:
            test = test.sort(self._get_order_of_sort())

        return test

    @abstractmethod
    def _core_split(self, log: AnyDataFrame) -> SplitterReturnType:
        """
        This method implements split strategy

        :param log: input DataFrame `[timestamp, user_id, item_id, relevance]`
        :returns: `train` and `test DataFrames
        """

    def split(self, log: AnyDataFrame) -> SplitterReturnType:
        """
        Splits input DataFrame into train and test

        :param log: input DataFrame ``[timestamp, user_id, item_id, relevance]``
        :returns: List of splitted DataFrames
        """
        train, test = self._core_split(log)
        test = self._drop_cold_items_and_users(train, test)

        return train, test
        # for i in range(1, len(res)):
        #     res[i] = self._drop_cold_items_and_users(
        #         res[0],
        #         res[i],
        #     )
        # return res

    def _recalculate_with_session_id_column(self, data: AnyDataFrame) -> AnyDataFrame:
        if isinstance(data, SparkDataFrame):
            return self._recalculate_with_session_id_column_spark(data)

        return self._recalculate_with_session_id_column_pandas(data)

    def _recalculate_with_session_id_column_pandas(self, data: PandasDataFrame) -> PandasDataFrame:
        agg_function_name = "first" if self.session_id_processing_strategy == "train" else "last"
        res = data.copy()
        res["is_test"] = res.groupby([self.user_col, self.session_id_col])["is_test"].transform(agg_function_name)

        return res

    def _recalculate_with_session_id_column_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        agg_function = sf.first if self.session_id_processing_strategy == "train" else sf.last
        res = data.withColumn(
            "is_test",
            agg_function("is_test").over(
                Window.orderBy(self.timestamp_col)
                .partitionBy(self.user_col, self.session_id_col)  # type: ignore
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            ),
        )

        return res
