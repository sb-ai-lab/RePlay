"""
These kind of splitters process log as a whole:

- by time
- at random by test size
- select cold users for test

"""

from datetime import datetime
from typing import Optional, Union

from pandas import DataFrame as PandasDataFrame
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame as SparkDataFrame, Window
from replay.utils.spark_utils import convert2spark

from replay.data import AnyDataFrame
from replay.splitters.base_splitter import (
    Splitter,
    SplitterReturnType,
)


# pylint: disable=too-few-public-methods
class DateSplitter(Splitter):
    """
    Split into train and test by date.
    """

    _init_arg_names = [
        "test_start",
        "drop_cold_users",
        "drop_cold_items",
        "drop_zero_rel_in_test",
        "user_col",
        "item_col",
        "timestamp_col",
        "rating_col",
        "session_id_col",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_start: Union[datetime, float, str, int],
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        drop_zero_rel_in_test: bool = True,
        user_col: str = "user_idx",
        item_col: Optional[str] = "item_idx",
        timestamp_col: Optional[str] = "timestamp",
        rating_col: Optional[str] = "relevance",
        session_id_col: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param test_start: string``yyyy-mm-dd``, int unix timestamp, datetime or a
            fraction for test size to determine the date automatically
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param drop_zero_rel_in_test: flag to remove entries with relevance <= 0
            from the test part of the dataset
        :param user_col: user id column name
        :param item_col: item id column name
        :param timestamp_col: timestamp column name
        :param rating_col: rating column name
        :param session_id_col: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            drop_zero_rel_in_test=drop_zero_rel_in_test,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            rating_col=rating_col,
            session_id_col=session_id_col,
            session_id_processing_strategy=session_id_processing_strategy
        )
        self.test_start = test_start

    def _get_order_of_sort(self) -> list:
        return [self.user_col, self.timestamp_col]

    def _core_split(self, log: AnyDataFrame) -> SplitterReturnType:
        if isinstance(self.test_start, float):
            dates = log.select(self.timestamp_col).withColumn(
                "_row_number_by_ts", sf.row_number().over(Window.orderBy(self.timestamp_col))
            )
            test_start = int(dates.count() * (1 - self.test_start)) + 1
            test_start = (
                dates.filter(sf.col("_row_number_by_ts") == test_start)
                .select(self.timestamp_col)
                .collect()[0][0]
            )
        else:
            dtype = dict(log.dtypes)[self.timestamp_col]
            test_start = sf.lit(self.test_start).cast(self.timestamp_col).cast(dtype)
        train = log.filter(sf.col(self.timestamp_col) < test_start)
        test = log.filter(sf.col(self.timestamp_col) >= test_start)
        return [train, test]


# pylint: disable=too-few-public-methods
class RandomSplitter(Splitter):
    """Assign records into train and test at random."""

    _init_arg_names = [
        "test_size",
        "drop_cold_users",
        "drop_cold_items",
        "drop_zero_rel_in_test",
        "seed",
        "user_col",
        "item_col",
        "timestamp_col",
        "rating_col",
        "session_id_col",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        drop_zero_rel_in_test: bool = True,
        seed: Optional[int] = None,
        user_col: str = "user_idx",
        item_col: Optional[str] = "item_idx",
        timestamp_col: Optional[str] = "timestamp",
        rating_col: Optional[str] = "relevance",
        session_id_col: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param test_size: test size 0 to 1
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param drop_zero_rel_in_test: flag to remove entries with relevance <= 0
            from the test part of the dataset
        :param seed: random seed
        :param user_col: user id column name
        :param item_col: item id column name
        :param timestamp_col: timestamp column name
        :param rating_col: rating column name
        :param session_id_col: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            drop_zero_rel_in_test=drop_zero_rel_in_test,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            rating_col=rating_col,
            session_id_col=session_id_col,
            session_id_processing_strategy=session_id_processing_strategy
        )
        self.seed = seed
        self.test_size = test_size
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must be 0 to 1")

    def _get_order_of_sort(self) -> list:
        pass

    def _core_split(self, log: AnyDataFrame) -> SplitterReturnType:
        train, test = convert2spark(log).randomSplit(
            [1 - self.test_size, self.test_size], self.seed
        )
        return [train, test]


# pylint: disable=too-few-public-methods
class NewUsersSplitter(Splitter):
    """
    Only new users will be assigned to test set.
    Splits log by timestamp so that test has `test_size` fraction of most recent users.


    >>> from replay.splitters import NewUsersSplitter
    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1,1,2,2,3,4],
    ...    "item_idx": [1,2,3,1,2,3],
    ...    "relevance": [1,2,3,4,5,6],
    ...    "timestamp": [20,40,20,30,10,40]})
    >>> data_frame
       user_idx  item_idx  relevance  timestamp
    0         1         1          1         20
    1         1         2          2         40
    2         2         3          3         20
    3         2         1          4         30
    4         3         2          5         10
    5         4         3          6         40
    >>> train, test = NewUsersSplitter(test_size=0.1).split(data_frame)
    >>> train.show()
    +--------+--------+---------+---------+
    |user_idx|item_idx|relevance|timestamp|
    +--------+--------+---------+---------+
    |       1|       1|        1|       20|
    |       2|       3|        3|       20|
    |       2|       1|        4|       30|
    |       3|       2|        5|       10|
    +--------+--------+---------+---------+
    <BLANKLINE>
    >>> test.show()
    +--------+--------+---------+---------+
    |user_idx|item_idx|relevance|timestamp|
    +--------+--------+---------+---------+
    |       4|       3|        6|       40|
    +--------+--------+---------+---------+
    <BLANKLINE>

    Train DataFrame can be drastically reduced even with moderate
    `test_size` if the amount of new users is small.

    >>> train, test = NewUsersSplitter(test_size=0.3).split(data_frame)
    >>> train.show()
    +--------+--------+---------+---------+
    |user_idx|item_idx|relevance|timestamp|
    +--------+--------+---------+---------+
    |       3|       2|        5|       10|
    +--------+--------+---------+---------+
    <BLANKLINE>
    """

    _init_arg_names = [
        "test_size",
        "drop_cold_users",
        "drop_cold_items",
        "drop_zero_rel_in_test",
        "user_col",
        "item_col",
        "timestamp_col",
        "rating_col",
        "session_id_col",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        drop_zero_rel_in_test: bool = True,
        user_col: str = "user_idx",
        item_col: Optional[str] = "item_idx",
        timestamp_col: Optional[str] = "timestamp",
        rating_col: Optional[str] = "relevance",
        session_id_col: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param test_size: test size 0 to 1
        :param drop_cold_items: flag to drop cold items from test
        :param drop_zero_rel_in_test: flag to remove entries with relevance <= 0
            from the test part of the dataset
        :param user_col: user id column name
        :param item_col: item id column name
        :param timestamp_col: timestamp column name
        :param rating_col: rating column name
        :param session_id_col: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            drop_zero_rel_in_test=drop_zero_rel_in_test,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            rating_col=rating_col,
            session_id_col=session_id_col,
            session_id_processing_strategy=session_id_processing_strategy
        )
        self.test_size = test_size
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must be 0 to 1")

    def _get_order_of_sort(self) -> list:
        pass

    def _core_split(self, log: AnyDataFrame) -> SplitterReturnType:
        start_date_by_user = log.groupby(self.user_col).agg(
            sf.min(self.timestamp_col).alias("_start_dt_by_user")
        )
        test_start_date = (
            start_date_by_user.groupby("_start_dt_by_user")
            .agg(sf.count(self.user_col).alias("_num_users_by_start_date"))
            .select(
                "_start_dt_by_user",
                sf.sum("_num_users_by_start_date")
                .over(Window.orderBy(sf.desc("_start_dt_by_user")))
                .alias("_cum_num_users_to_dt"),
                sf.sum("_num_users_by_start_date").over(Window.orderBy(sf.lit(1))).alias("total"),
            )
            .filter(sf.col("_cum_num_users_to_dt") >= sf.col("total") * self.test_size)
            .agg(sf.max("_start_dt_by_user"))
            .head()[0]
        )

        train = log.filter(sf.col(self.timestamp_col) < test_start_date)

        test = log.join(
            start_date_by_user.filter(sf.col("_start_dt_by_user") >= test_start_date),
            how="inner",
            on=self.user_col,
        ).drop("_start_dt_by_user")
        return [train, test]


# pylint: disable=too-few-public-methods
class ColdUserRandomSplitter(Splitter):
    """
    Test set consists of all actions of randomly chosen users.
    """

    # для использования в тестах

    _init_arg_names = [
        "test_size",
        "drop_cold_users",
        "drop_cold_items",
        "drop_zero_rel_in_test",
        "seed",
        "user_col",
        "item_col",
        "timestamp_col",
        "session_id_col",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        drop_zero_rel_in_test: bool = True,
        seed: Optional[int] = None,
        user_col: str = "user_idx",
        item_col: Optional[str] = "item_idx",
        timestamp_col: Optional[str] = "timestamp",
        rating_col: Optional[str] = "relevance",
        session_id_col: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param test_size: fraction of users to be in test
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param drop_zero_rel_in_test: flag to remove entries with relevance <= 0
            from the test part of the dataset
        :param seed: random seed
        :param user_col: user id column name
        :param item_col: item id column name
        :param timestamp_col: timestamp column name
        :param rating_col: rating column name
        :param session_id_col: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            drop_zero_rel_in_test=drop_zero_rel_in_test,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            rating_col=rating_col,
            session_id_col=session_id_col,
            session_id_processing_strategy=session_id_processing_strategy
        )
        self.seed = seed
        self.test_size = test_size
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must be 0 to 1")
        
    def _get_order_of_sort(self) -> list:
        pass

    def _core_split(self, log: AnyDataFrame) -> SplitterReturnType:
        log = convert2spark(log)
        users = log.select(self.user_col).distinct()
        train_users, test_users = users.randomSplit(
            [1 - self.test_size, self.test_size],
            seed=self.seed,
        )
        train = log.join(train_users, on=self.user_col, how="inner")
        test = log.join(test_users, on=self.user_col, how="inner")
        return [train, test]
